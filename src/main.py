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
from fastapi.responses import StreamingResponse
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
        if not isinstance(doc, dict):
            continue
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
    log_raw = data.get("metadata_log", {}) if isinstance(data, dict) else {}
    log = log_raw if isinstance(log_raw, dict) else {}
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
    """Rebuild the sources list from the last successful retrieve_docs ToolMessage."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            sources = _sources_from_tool_content(msg.content)
            if sources:
                return sources
    return []


def _build_config() -> dict:
    """Build the per-request LangGraph config (Langfuse handler + session id)."""
    session_id = str(uuid.uuid4())
    return {
        "configurable": {},
        "recursion_limit": 20,
        "callbacks": [CallbackHandler()],
        "metadata": {
            "session_id": session_id,
            "langfuse_trace_name": "api_ask_agent",
        },
    }


@app.post("/ask")
async def ask_agent(payload: AskRequest):
    question = payload.question
    try:
        config = _build_config()

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


_GEN_LABELS = {
    "generator": "✍️ Redactando respuesta",
    "listing_generator": "🗂️ Armando el listado",
}


@app.post("/ask/stream")
async def ask_agent_stream(payload: AskRequest):
    config = _build_config()
    inputs = {"messages": [HumanMessage(content=payload.question)]}

    async def event_stream():
        streamed_token = False
        gen_step_sent = False
        last_tool_content = None
        try:
            async for mode, chunk in agent_graph.astream(
                inputs, config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node_name, update in chunk.items():
                        if node_name == "tools":
                            msgs = update.get("messages", [])
                            if msgs:
                                last_tool_content = msgs[-1].content

                        step = build_step(node_name, update)
                        if step:
                            yield _format_sse(step)

                        # No-token generators (listing / "no documents" fallback):
                        # emit their step + full content here.
                        if node_name in _GEN_LABELS and not streamed_token:
                            if not gen_step_sent:
                                yield _format_sse({
                                    "type": "step", "id": node_name,
                                    "label": _GEN_LABELS[node_name], "detail": None,
                                })
                                gen_step_sent = True
                            msgs = update.get("messages", [])
                            text = getattr(msgs[-1], "content", "") if msgs else ""
                            if text:
                                yield _format_sse({"type": "token", "text": text})

                elif mode == "messages":
                    msg_chunk, metadata = chunk
                    node = metadata.get("langgraph_node")
                    if node in _GEN_LABELS:
                        if not gen_step_sent:
                            yield _format_sse({
                                "type": "step", "id": node,
                                "label": _GEN_LABELS[node], "detail": None,
                            })
                            gen_step_sent = True
                        text = getattr(msg_chunk, "content", "")
                        if text:
                            streamed_token = True
                            yield _format_sse({"type": "token", "text": text})

            sources = _sources_from_tool_content(last_tool_content) if last_tool_content else []
            yield _format_sse({"type": "sources", "sources": sources})
            yield _format_sse({"type": "done"})
        except Exception as e:
            print(f"--- [STREAM ERROR] {e} ---")
            yield _format_sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
