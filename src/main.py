import json
import sys
import uuid
from contextlib import asynccontextmanager

# En Windows la consola usa cp1252 por defecto. Los print() de los nodos del
# agente incluyen tildes y emojis (⚡, ⚠️, 🛑, ✅, 👤) que NO se pueden codificar
# en cp1252: eso lanza UnicodeEncodeError y tumba el request (sobre todo el path
# LISTING). Forzamos UTF-8 en stdout/stderr para que el logging sea seguro.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage

# 1. Importar el manejador de Langfuse para LangChain
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from agent.graph import app as agent_graph
import os

os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "development"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nada especial por ahora.
    yield
    # Shutdown: aseguramos que los trazos pendientes se envíen a Langfuse.
    # Aunque el SDK v3 maneja el vaciado en hilos de fondo, esto evita
    # perder datos si el proceso muere abruptamente.
    Langfuse().flush()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


def _val(value):
    """Return the .value of an Enum, otherwise the value unchanged."""
    return getattr(value, "value", value)


def _format_sse(payload: dict) -> str:
    """Serialize a payload as a single Server-Sent Event message."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sources_from_tool_content(content) -> list:
    """Parse the sources list from a single retrieve_docs ToolMessage payload."""
    if not isinstance(content, str):
        return []
    if content.startswith("LOW_RELEVANCE_ERROR") or content.startswith("No documents found"):
        return []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    documents = data.get("documents") if isinstance(data, dict) else None
    if not documents:
        return []

    sources, seen = [], set()
    for doc in documents:
        url = doc.get("url")
        if not url or url in {"No URL", "#"} or url in seen:
            continue
        sources.append({
            "title": doc.get("title") or "Bitovi Blog Post",
            "url": url,
            "author": doc.get("author") or "Bitovi Expert",
        })
        seen.add(url)
    return sources


def _tool_step_detail(update: dict):
    """Build the technical detail for the 'tools' step from its ToolMessage."""
    messages = update.get("messages", [])
    if not messages:
        return None
    content = messages[-1].content
    if isinstance(content, str) and content.startswith("LOW_RELEVANCE_ERROR"):
        return {"estado": "relevancia baja, se expandirá la query"}
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    log = data.get("metadata_log", {}) if isinstance(data, dict) else {}
    return {
        "estrategia": log.get("strategy"),
        "documentos": log.get("count"),
        "orden": log.get("sort_by"),
    }


def build_step(node_name: str, update: dict):
    """Map a completed graph node + its state update to a friendly UI step.

    Returns None for internal/router nodes and for the generator nodes
    (whose steps are emitted by the streaming loop alongside the first token).
    """
    if node_name == "intent_analyzer":
        return {
            "type": "step",
            "id": node_name,
            "label": "🔍 Entendiendo tu pregunta",
            "detail": {
                "intencion": _val(update.get("task_type")),
                "razonamiento": update.get("intent_reasoning"),
            },
        }
    if node_name == "analizar":
        params = update.get("search_params", {}) or {}
        return {
            "type": "step",
            "id": node_name,
            "label": "🧭 Eligiendo estrategia",
            "detail": {
                "ruta": _val(update.get("selected_route")),
                "sort_by": params.get("sort_by"),
                "top_k": params.get("top_k"),
            },
        }
    if node_name == "query_optimizer":
        return {
            "type": "step",
            "id": node_name,
            "label": "✨ Optimizando la búsqueda",
            "detail": {"query": update.get("search_query")},
        }
    if node_name == "tools":
        return {
            "type": "step",
            "id": node_name,
            "label": "📚 Buscando en el blog",
            "detail": _tool_step_detail(update),
        }
    if node_name == "expansion":
        retries = update.get("retry_count", 0)
        return {
            "type": "step",
            "id": node_name,
            "label": f"🔁 Refinando la búsqueda (intento {retries})",
            "detail": {"query": update.get("search_query")},
        }
    return None


def _extract_sources(messages) -> list:
    """
    Reconstruye la lista de fuentes a partir del último ToolMessage exitoso.

    La tool `retrieve_docs` devuelve un JSON con la forma:
        {"metadata_log": {...}, "documents": [{"title", "author", "url", "year", ...}]}

    El estado del agente NO expone los Document originales, así que parseamos
    ese payload para alimentar la sección "Fuentes Citadas" del frontend.
    """
    sources_list = []
    seen_urls = set()

    # Buscamos el último ToolMessage que sea un payload válido (no un error).
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content
        if not isinstance(content, str):
            continue
        if content.startswith("LOW_RELEVANCE_ERROR") or content.startswith("No documents found"):
            continue

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        documents = data.get("documents") if isinstance(data, dict) else None
        if not documents:
            continue

        for doc in documents:
            url = doc.get("url")
            if not url or url in {"No URL", "#"} or url in seen_urls:
                continue
            sources_list.append({
                "title": doc.get("title") or "Bitovi Blog Post",
                "url": url,
                "author": doc.get("author") or "Bitovi Expert",
            })
            seen_urls.add(url)

        # Solo usamos el último retrieval exitoso.
        break

    return sources_list


@app.post("/ask")
async def ask_agent(payload: AskRequest):
    question = payload.question
    try:
        session_id = str(uuid.uuid4())

        # 2. Inicializar el CallbackHandler de Langfuse.
        # El SDK v3 detecta automáticamente LANGFUSE_PUBLIC_KEY,
        # LANGFUSE_SECRET_KEY y LANGFUSE_HOST desde el entorno.
        langfuse_handler = CallbackHandler()

        # El agente es single-turn: no usamos checkpointer, por lo tanto
        # no pasamos thread_id (sería inerte). session_id queda en metadata
        # solo para correlacionar trazas en Langfuse.
        config = {
            "configurable": {},
            "recursion_limit": 20,
            "callbacks": [langfuse_handler],
            "metadata": {
                "session_id": session_id,
                "langfuse_trace_name": "api_ask_agent",
            },
        }

        inputs = {
            "messages": [HumanMessage(content=question)],
        }

        result = await agent_graph.ainvoke(inputs, config=config)

        messages = result.get("messages", [])
        if not messages:
            return {"error": "El agente no generó mensajes de respuesta."}

        final_answer = messages[-1].content
        sources_list = _extract_sources(messages)

        return {"response": final_answer, "sources": sources_list}

    except Exception as e:
        print(f"--- [ERROR] {str(e)} ---")
        return {"error": str(e)}
