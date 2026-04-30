# retrieve_docs
# web_search
from dotenv import load_dotenv
load_dotenv()


import os
from langchain_core.tools import tool
from scripts import utils
from langgraph.prebuilt import InjectedState
from typing import Annotated
import json
from scripts.schemas import TaskType
from agent.state import AgentState
from langchain_core.runnables import RunnableConfig



@tool
def retrieve_docs(query: str, state: Annotated[dict, InjectedState],  k: int = 10):
    """
    Recupera documentos técnicos de la base de conocimientos.
    Args:
        query: La consulta de búsqueda optimizada.
        k: Cantidad de documentos a recuperar.
    """
    # 1. Extraemos el estado del agente desde el config inyectado por LangGraph
    # El estado vive en 'configurable'
    
    
    # 2. Tomamos 'selected_route' directamente del estado
    # Ya no dependemos de que el LLM lo pase como argumento
    selected_route = state.get("selected_route")
    params = state.get("search_params")
    original_query = state['messages'][0].content
    
    print(f"\n[TOOL] retrieve_docs invocado")
    print(f" -> Query: {query}")
    print(f" -> Route desde el Estado: {selected_route}")
    print(f" -> Parametros: {params}")
       
    print(f"[TOOL] Ejecutando búsqueda para ruta: {selected_route}")
    
    sort_by = params.get("sort_by", "relevance")
    top_k = params.get("top_k", 5)    
    k = max(k, top_k)


    print(f"[TOOL] Usando parámetros del Analyzer -> Route: {selected_route}, Sort: {sort_by}, TopK: {top_k}")
    
    # 2. Obtener filtros (Esto sí depende de la query específica)
    print(f"[TOOL] Extrayendo metadata de la query original")
    filters = utils.extract_filters(original_query)
    print(f"[TOOL] Filters: {filters}")    

    # --- ESTRATEGIA DE BÚSQUEDA ---
    if selected_route.lower() == "fast":
        print("[MODE] Fast Retrieval - Skipping semantic")
        docs = utils.get_docs_by_metadata(filters, k=k)
    else:
        ranking_keywords = utils.generate_ranking_keywords(query)
        print(f"[MODE] Convencional - Ranking Keywords: {ranking_keywords}")
        # Fetch docs for re-ranking
        results = utils.search_docs(query, filters, ranking_keywords, k=k*5)
        if not results or not ranking_keywords:
            print("[MODE] Skipping Re-rank: No keywords found")
            docs = results
            scores = [2.0] * len(results) # Scores por defecto
        else:
            results = utils.rank_documents_by_keywords(results, ranking_keywords, k=k)        
            docs = results['docs']
            scores = results['scores']
            best_score = scores[0] if scores else 0
            THRESHOLD = 1.0
            if best_score < THRESHOLD:
                print(f"[TOOL] LOW_RELEVANCE_ERROR: The best document score is only {best_score:.2f}. Need query expansion.")
                return f"LOW_RELEVANCE_ERROR: The best document score is only {best_score:.2f}. Need query expansion."

    if not docs:
        return f"No documents found for the query: '{query}'."

    # --- POST-PROCESS SEGURO (Aquí usamos los parámetros del Analyzer) ---
    print(f"[TOOL] Aplicando post-proceso: Sort by {sort_by}, Limit {top_k}")
    docs = utils.process_results(docs, sort_by=sort_by, k=top_k)
    
# --- REEMPLAZA EL FINAL DE TU TOOL ---
    
   # 3. FORMATEO PARA EL AGENTE
    # Extraemos el intent del estado (asegúrate de que 'state' esté disponible en el scope)
    intent_type = state.get("intent") 
    
    output_data = {
        "metadata_log": {
            "strategy": selected_route.upper(),
            "sort_by": sort_by,
            "count": len(docs)
        },
        "documents": []
    }

    for doc in docs:
        meta = doc.metadata
        
        # Construimos el diccionario del documento
        doc_entry = {
            "title": meta.get('title') or "Untitled Document",
            "author": meta.get('author') or "Unknown Author",
            "url": meta.get('source') or meta.get('url') or "No URL",
            "year": meta.get('year') or "N/A"
        }
        
        # CONDICIONAL CRÍTICO: 
        # Si NO es listing, incluimos el contenido para análisis (RAG/QA)
        # Si ES listing, dejamos el contenido vacío para ahorrar tokens y evitar resúmenes
        if intent_type != TaskType.LISTING:
            doc_entry["content"] = doc.page_content
        else:
            # Opcional: puedes no incluir la llave 'content' directamente
            doc_entry["content"] = "" 

        output_data["documents"].append(doc_entry)
    
    return json.dumps(output_data)

    # # 3. FORMATEO PARA EL AGENTE
    # retrieved_sections = []
    # for i, doc in enumerate(docs, 1):
    #     meta = doc.metadata
    #     m_title = meta.get('title') or "Untitled Document"
    #     m_author = meta.get('author') or "Unknown Author"
    #     m_url = meta.get('source') or meta.get('url') or "No URL"
    #     m_year = meta.get('year') or "N/A"

    #     doc_block = (
    #         f"--- Document {i} ---\n"
    #         f"REFERENCE_TITLE: {m_title}\n"
    #         f"REFERENCE_AUTHOR: {m_author}\n"
    #         f"REFERENCE_YEAR: {m_year}\n"
    #         f"REFERENCE_URL: {m_url}\n"
    #         f"CONTENT:\n{doc.page_content}\n"
    #         f"-------------------"
    #     )
    #     retrieved_sections.append(doc_block)

    # header = (
    #     f"--- INTERNAL SEARCH LOG ---\n"
    #     f"Used Strategy: {selected_route.upper()}\n"
    #     f"Sorted By: {sort_by}\n"
    #     f"Docs Delivered: {len(docs)}\n"
    #     f"---------------------------\n\n"
    # )
    
    # return header + "\n\n".join(retrieved_sections)

# tavily search dev serper, ddgs
from ddgs import DDGS

@tool
def web_search(query:str, num_results: int = 10) -> str:
    """Use this tool whenever you need to access realtime or latest information.
        Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
    
    Returns:
        Formatted search results with titles, descriptions, and URLs
    """

    results = DDGS().text(query=query, max_results=num_results, region='us-en')

    if not results:
        return f"No results found for '{query}'"
    
    formatted_results = [f"Search results for search query: '{query}'"]
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        href = result.get('href', '')
        body = result.get('body', 'No description available')
        
        text = f"{i}. **{title}**\n   {body}\n   {href}"
        
        formatted_results.append(text)

    return "\n\n".join(formatted_results)