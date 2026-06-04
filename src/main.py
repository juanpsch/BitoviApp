import json
import uuid
from contextlib import asynccontextmanager

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
