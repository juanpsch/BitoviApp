import json
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

# 1. Importar el manejador de Langfuse para LangChain
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from agent.graph import app as agent_graph
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "development"

@app.get("/ask")
async def ask_agent(question: str):
    try:
        session_id = str(uuid.uuid4())

        # 2. Inicializar el CallbackHandler de Langfuse.
        # El SDK v3 detectará automáticamente las variables de entorno:
        # LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        langfuse_handler = CallbackHandler()

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 20,
            # 3. Pasar el handler dentro de la lista de callbacks
            "callbacks": [langfuse_handler],
            "metadata": {
                "session_id": session_id,
                "langfuse_trace_name": "api_ask_agent"                                
            }
        }

        inputs = {
            "messages": [HumanMessage(content=question)],
            "retrieved_docs": [],
            "current_step_idx": 0,
        }

        # Al usar ainvoke con los callbacks en la config, LangGraph
        # reportará automáticamente cada nodo, pasos y llamadas a LLMs.
        result = await agent_graph.ainvoke(inputs, config=config)

        if "messages" in result and len(result["messages"]) > 0:
            final_answer = result["messages"][-1].content

            raw_docs = result.get("retrieved_docs", [])
            sources_list = []
            seen_urls = set()

            for d in raw_docs:
                url = d.metadata.get("source") or d.metadata.get("url")
                title = d.metadata.get("title") or "Bitovi Blog Post"
                author = d.metadata.get("author") or "Bitovi Expert"

                if url and url not in seen_urls:
                    sources_list.append(
                        {"title": title, "url": url, "author": author}
                    )
                    seen_urls.add(url)

            return {"response": final_answer, "sources": sources_list}
        else:
            return {"error": "El agente no generó mensajes de respuesta."}

    except Exception as e:
        print(f"--- [ERROR] {str(e)} ---")
        return {"error": str(e)}


# 4. Asegurar que los trazos pendientes se envíen al apagar la API
@app.on_event("shutdown")
async def shutdown_event():
    # Aunque el SDK v3 maneja el vaciado en hilos de fondo,
    # esto asegura que no se pierda nada si el proceso muere bruscamente.
    from langfuse import Langfuse

    Langfuse().flush()