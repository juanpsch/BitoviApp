import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.graph import app as agent_graph
from langchain_core.messages import HumanMessage
import uuid  # Importante para generar IDs únicos

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
async def ask_agent(question: str):
    try:
        session_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 20
        }

        inputs = {
            "messages": [HumanMessage(content=question)],
            "retrieved_docs": [],
            "current_step_idx": 0
        }
        
        result = await agent_graph.ainvoke(inputs, config=config)
        
        if "messages" in result and len(result["messages"]) > 0:
            final_answer = result["messages"][-1].content
            
            # --- MEJORA AQUÍ ---
            # Extraemos objetos completos de fuentes para que el Front los use
            raw_docs = result.get("retrieved_docs", [])
            sources_list = []
            seen_urls = set()

            for d in raw_docs:
                url = d.metadata.get('source') or d.metadata.get('url')
                title = d.metadata.get('title') or "Bitovi Blog Post"
                author = d.metadata.get('author') or "Bitovi Expert"
                
                if url and url not in seen_urls:
                    sources_list.append({
                        "title": title,
                        "url": url,
                        "author": author
                    })
                    seen_urls.add(url)

            return {
                "response": final_answer,
                "sources": sources_list  # Ahora enviamos objetos, no solo strings
            }
        else:
            return {"error": "El agente no generó mensajes de respuesta."}

    except Exception as e:
        print(f"--- [ERROR] {str(e)} ---")
        return {"error": str(e)}