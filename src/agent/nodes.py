from .state import AgentState
from scripts import utils
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.graph import END, START
from config import llm
from scripts.my_tools import retrieve_docs
from scripts.schemas import BlogCategory
from scripts.schemas import IntentOutput
from scripts.schemas import TaskType, SearchQueryOutput
import json
from scripts.mapping import GLOSARIO_ACRONIMOS

# =============================================================================
# Pydantic Schemas for Structured Outputs
# =============================================================================
from pydantic import BaseModel, Field

# =============================================================================
# Agent Node
# =============================================================================



def intent_analyzer_node(state: AgentState):
    print("\n[INTENT_ANALIZER-NODE] Clasificando Intención del Usuario")
    
    intent_prompt = """Analyze the user query and determine the task type based on the **Content vs. Container** logic.

### 1. SINTESIS (Concept & Content Research)
- **Goal**: To understand "What", "How", or "Why".
- **Rule**: Use this for ANY conceptual definition, explanation of a process, or specific data extraction.
- **Keywords**: "What is...", "How does...", "Explain...", "What is the meaning of...".
- **CRITICAL**: "What is a RAG?" or "What is a vector store?" are ALWAYS SINTESIS. The user wants a definition, not a list of files.

### 2. LISTING (Directory & Discovery)
- **Goal**: To see a list of available resources or documents, blogs,. artticles or posts.
- **Rule**: Use this ONLY when the user wants to see "What articles exist" or "Give me the links/titles".
- **Keywords**: "List all...", "Show me blogs about...", "Which posts...".

### 3. REASONING (Deep Analysis)
- **Goal**: To compare or evaluate trade-offs.
- **Rule**: Use this for "Compare X vs Y", "Pros and Cons", or "Why is X better than Y".

### ANTI-HALLUCINATION & LOGIC RULES:
* **NO ANSWERING**: Do not answer the question in the reasoning field. Only explain why you chose the category.
* **DEFINITION > LIST**: If the user asks for a definition of a technical term (even if it's an acronym like RAG), it is SINTESIS.
* **CONTENT OVER CONTAINER**: If the user asks "What is [Article] about?", they want the content (SINTESIS), not the list entry (LISTING).

### THE "CONTENT OF" RULE:
- If the user asks "What is [X] about?", "Summarize [X]", or "What does [X] say?" -> **Intent: SINTESIS**.
- Even if [X] is a "document", "article", or "blog", the intent is to extract KNOWLEDGE from it, not just to find the file.

### REFINED CATEGORIES:
- **LISTING**: Use this ONLY when the user wants to SEE a list or FIND which documents exist (e.g., "Show me the last 5 articles").
- **SINTESIS**: Use this when the user wants to UNDERSTAND the information inside a document (e.g., "What is the last article about?").

### EXAMPLES:
- "Show me the latest article" -> LISTING (The goal is to get the link/title).
- "What is the latest article about?" -> SINTESIS (The goal is to understand the topic).
- "List Bitovi articles" -> LISTING.
- "Explain Bitovi's latest DevOps post" -> SINTESIS.
- "What tools does Bitovi recommend for E2E?" -> SINTESIS (Investigating internal recommendations).
- "Show me all articles about E2E tools" -> LISTING (Retrieving the document containers).


"""
    
    # Forzamos la salida estructurada
    structured_llm = llm.with_structured_output(IntentOutput)
    
    # Analizamos el primer mensaje del usuario
    user_query = state['messages'][0].content
    analysis = structured_llm.invoke([
        SystemMessage(content=intent_prompt),
        HumanMessage(content=user_query)
    ])
    
    print(f"[INTENT_ANALIZER-NODE] Type: {analysis.intent} | Reason: {analysis.reasoning}")
    
    # Guardamos la intención en el estado para que los siguientes nodos la vean
    return {"task_type": analysis.intent}

def analysis_node(state: AgentState):
    """
    Nodo encargado de pre-procesar la query del usuario.
    """
    user_query = state['messages'][-1].content
    
    print(f"[ANALYSIS-NODE] Analizando query: {user_query}")
    
    # 1. Ejecutamos tus funciones de Utils (Python puro)
    strategy_result = utils.define_retrieval_strategy(user_query)
    control_result = utils.analyze_search_control(user_query)
    
    # 2. Preparamos el diccionario de parámetros
    # Asumimos que analyze_search_control devuelve un objeto con sort_by y top_k
    search_params = {
        "sort_by": control_result.sort_by,
        "top_k": control_result.top_k
    }
    
    print(f"[ANALYSIS-NODE] Route: {strategy_result.route} | Params: {search_params}")

    # 3. Retornamos la actualización del estado
    return {
        "selected_route": strategy_result.route,
        "search_params": search_params
    }

def query_optimizer_node(state: AgentState):
    print("\n[QUERY_OPTIMIZER-NODE] Optimizando con Glosario")
    
    task = state.get('task_type') 
    user_query = state['messages'][0].content

    # Preparamos el glosario como una lista de ejemplos (Few-Shot implícito)
    glosario_string = "\n".join([f"- {k} => {v}" for k, v in GLOSARIO_ACRONIMOS.items()])    

    # 1. Definimos el rol y la base de conocimientos
    system_instruction = (
        "You are a technical search expert. You have a GLOSSARY of acronyms. "
        "Your task is to REWRITE the user query into high-relevance search terms.\n\n"
        "GLOSSARY:\n"
        f"{glosario_string}"
    )

    # 2. Definimos las reglas tácticas
    user_instruction = f"""TRANSFORMATION RULES:
    1. If you see an acronym (like AI, IA, RAG), you MUST replace it with its FULL MEANING from the glossary.
    2. Add the original acronym next to the full meaning.
    3. Task Context ({task}): 
       - If 'listing': Use nouns and technologies.
       - If 'synthesis': Use conceptual/architectural terms.
    4. NO 'Bitovi', NO 'articles', NO 'posts'.
    5. ONLY output the keywords.

    USER QUERY TO TRANSFORM: {user_query}
    """
    
    # 3. Usamos la estructura de mensajes para dar más peso a las instrucciones
    structured_llm = llm.with_structured_output(SearchQueryOutput)
    
    # El truco está en separar la instrucción del glosario del mensaje del usuario
    result = structured_llm.invoke([
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction}
    ])
    
    # Post-proceso manual de seguridad (Opcional pero recomendado para 'AI')
    final_query = result.search_query.strip()
    if final_query.upper() == "AI" or final_query.upper() == "IA":
        final_query = "Artificial Intelligence LLM RAG"

    print(f"[OPTIMIZER] Result: {final_query}")
    
    return {"search_query": final_query}

def grade_retrieval(state: AgentState):
    """
    Actúa como un 'Quality Gate'. Evalúa si el resultado de la tool 
    es lo suficientemente bueno para pasar al generador.
    """
    last_message = state['messages'][-1].content
    retries = state.get("retry_count", 0)

    # Si detectamos el error de score bajo que definimos en la tool
    if "LOW_RELEVANCE_ERROR" in last_message:
        if retries < 3:
            print(f"⚠️ Calidad insuficiente (Score < theshold). Iniciando expansión {retries + 1}/3")
            return "expand"
        else:
            print("🛑 Umbral no alcanzado tras 3 intentos. Forzando finalización.")
            return "fail"
    
    # Si no hay error, los documentos son buenos.
    return "end"


def retrieval_node(state: AgentState):
    optimized_query = state.get('search_query')
    route = state.get('selected_route', 'CONVENCIONAL')
    params = state.get('search_params', {})
    requested_k = params.get('top_k', 10)

    # 1. Forzamos la llamada a la tool específica
    llm_with_forced_tool = llm.bind_tools([retrieve_docs], tool_choice="retrieve_docs")

    # 2. Instrucciones ultra-precisas
    prompt = f"""Estrategia de búsqueda: {route}.
    Busca información para la siguiente consulta: {optimized_query}
    Parámetro K: {requested_k}
    
    REGLA: Invoca la herramienta 'retrieve_docs' con estos parámetros."""

    # 3. Invocación limpia
    # Pasamos el estado en el config para que la tool pueda leerlo si lo necesita
    response = llm_with_forced_tool.invoke(
        [SystemMessage(content="Eres un buscador técnico."), HumanMessage(content=prompt)],
        config={"configurable": {"state": state}} 
    )

    return {"messages": [response]}



import re

def expansion_node(state: AgentState):
    original_message = state['messages'][0]
    retries = state.get("retry_count", 0)

    # 1. Obtener la query base de forma segura (asegurando que sea string)
    raw_query = state.get('search_query') or original_message.content
    optimized_query = raw_query.content if hasattr(raw_query, 'content') else str(raw_query)

    print(f"[EXPANSION-NODE] Expandiendo query (Intento {retries + 1})")
    
    tech_rules = "\n".join([f"- {c.tech_mapping} -> {c.value}" for c in BlogCategory])
    
    # 2. Prompt con instrucción de formato negativa (para evitar el "Here are...")
    system_prompt = (
        "You are a technical search expert. Expand the query using these rules:\n"
        f"{tech_rules}\n"
        "STRICT RULE: Output ONLY technical keywords separated by spaces. "
        "No introductory text, no numbers, no bullet points, no markdown, no quotes. "
        "Limit to the 5 most important terms. Do not repeat words from the query."
    )
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": optimized_query}
    ])

    # 3. Limpieza agresiva de la respuesta del LLM (post-processing)
    res_text = response.content.strip()
    
    # Eliminamos cualquier línea que contenga ":" (introducciones como "Keywords:")
    clean_lines = [line for line in res_text.split('\n') if ':' not in line]
    clean_text = " ".join(clean_lines)
    
    # Eliminamos numeraciones (1. , 2.), guiones, comillas y backticks
    clean_text = re.sub(r'\d+\.\s*|-\s*|[`"\'“”]', '', clean_text)
    
    # 4. Construcción y deduplicación
    # Usamos la query original + la expansión para evitar el crecimiento infinito
    # pero manteniendo el contexto técnico.
    combined_content = f"{optimized_query} {clean_text}"
    unique_terms = list(dict.fromkeys(combined_content.split()))
    new_content = " ".join(unique_terms)

    print(f"[EXPANSION-NODE] Nueva Query Saneada: {new_content}")   
    
    return {
        "search_query": new_content,
        "retry_count": retries + 1    
    }

def generator_node(state: AgentState):
    print("\n--- [GENERATOR-NODE] Generando respuesta final ---")
    
    task_type = state.get("task_type")

    # El System Prompt con todas tus reglas de formato

    # 2. Definimos instrucciones dinámicas según el TaskType

    if task_type == TaskType.REASONING:
        intent_instructions = """
        ### USER INTENT: ANALYSIS & REASONING
        - Compare the different perspectives found in the documents.
        - Highlight trade-offs, pros/cons, and specific technical arguments.
        - Use a structured comparison format (headings or tables).
        """

    else: # SINTESIS
        intent_instructions = """
        ### USER INTENT: KNOWLEDGE EXTRACTION / SYNTHESIS
        - Provide a clear and detailed answer based on the internal content of the documents.
        - If the user asks "what is this about?", summarize the core message of the text.
        - Focus on "How-to" steps, definitions, or technical recommendations.
        """

    # 3. Construimos el System Prompt Final
    gen_prompt = f"""You are a specialized Tech Assistant at Bitovi. 
CRITICAL RULE: Your answer must be based EXCLUSIVELY on the provided context.

- If the information is not present in the documents, explicitly state: "I don't have articles in the Bitovi blog regarding this specific topic."
- DO NOT use external knowledge or provide external links (like official documentation) unless they are explicitly listed in the provided documents.
- Every claim must be backed by a document from the context.

    {intent_instructions}
    
    ### ANSWER FORMATTING:
    - Use **headings** (##, ###) for sections.
    - Use paragraphs for detailed findings and reasonings.
    - Use **bullet points** for lists.
    - Use **tables** for comparisons and structured data.
    - Use **bold** for emphasis on key metrics.
    - Cite sources: (Author: X, Year: Y, Title: Z, Source: url link)

    
    ### SOURCES SECTION:
        At the very end of your response, create a section titled "## Sources".
        For each document used, follow this format:
        - **Title**: [Title of the Document]
        - **Author**: [Author] | **Year**: [Year]
        - **Link**: [Click here to read](URL)
---

   
    If the context doesn't have the answer, be honest and say so."""

    # 1. Recuperamos la pregunta original (el primer mensaje)
    user_question = state['messages'][0] 
    
    # 2. Recuperamos SOLO los documentos de la última búsqueda exitosa
    # Buscamos el último ToolMessage que NO sea un error
    all_tool_messages = [m for m in state['messages'] if isinstance(m, ToolMessage)]
    
    # Si no hay mensajes de tool exitosos, esto va a fallar, así que validamos:
    if not all_tool_messages:
        return {"messages": [AIMessage(content="Lo siento, no pude encontrar información relevante.")]}
    
    last_docs = all_tool_messages[-1] # El último es el que tuvo score 4.04

    # 3. Construimos un historial limpio para el LLM
    clean_messages = [
        SystemMessage(content=gen_prompt),
        user_question, # La pregunta original (Whats a RAG?)
        last_docs      # Los documentos encontrados
    ]
    
    # 4. Invocamos
    response = llm.invoke(clean_messages)
    
    return {"messages": [response]}




def listing_generator_node(state: AgentState):
    print("\n[LISTING_GENERATOR] ⚡ Renderizado Determinístico (Clean UI)")
    
    messages = state.get('messages', [])
    tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    
    if not tool_msg:
        return {"messages": [AIMessage(content="No se encontraron datos.")]}
    
    try:
        data = tool_msg.content
        while isinstance(data, str):
            data = json.loads(data)
        articles = data.get("documents", [])
    except:
        return {"messages": [AIMessage(content="Error de formato en los artículos.")]}

    cards = []
    for art in articles:
        # 1. Extracción y Limpieza de Título
        raw_title = art.get('title') or art.get('name')
        if not raw_title or raw_title == "Untitled Document":
            continue
            
        # Limpiamos el título de ruidos (fechas, párrafos pegados, etc.)
        # Si hay más de dos espacios seguidos, probablemente empezó el contenido del blog
        clean_title = re.split(r'\s{2,}', str(raw_title))[0]
        # Quitamos la mención a Bitovi para cumplir con tu restricción
        clean_title = clean_title.replace("Bitovi", "").strip()
        
        # 2. Otros Metadatos
        url = art.get('url', '#')
        author = art.get('author', 'Unknown Author')
        year = art.get('year', 'N/A')
        
        # 3. Construcción de la Card
        card = f"## [{clean_title}]({url})\n*Author: {author}* | *Year: {year}*\n\n---\n"
        cards.append(card)

    # --- RESPUESTA FINAL SINCRONIZADA ---
    real_total = len(cards)
    if real_total == 0:
        return {"messages": [AIMessage(content="No se encontraron artículos con formato válido.")]}
    
    header = f"He encontrado **{real_total}** artículos relacionados:\n\n"
    return {"messages": [AIMessage(content=header + "".join(cards))]}



def listing_generator_node2(state: AgentState):
    print("\n[LISTING_GENERATOR-NODE] Listing Generator (Visual Polish)")
    
    messages = state.get('messages', [])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    
    if not tool_messages:
        return {"messages": [SystemMessage(content="No articles found.")]}
    
    raw_data = tool_messages[-1].content
    
    # --- NUEVA LÓGICA DE CONTEO ---
    try:
        data_parsed = json.loads(raw_data)
        count = len(data_parsed) if isinstance(data_parsed, list) else 0
    except:
        count = 0

    user_query = messages[0].content.lower()
    # Detectamos si la intención es contar
    pide_conteo = any(w in user_query for w in ["cuántos", "how many", "count", "cantidad", "total"])

    if pide_conteo:
        # Prompt específico para contar sin romper el estilo
        system_prompt = f"""You are a Technical Librarian.
        The user wants to know the total number of items.
        I found exactly {count} articles.
        
        TASK: State the total count clearly and professionally.
        - NO introductory fluff.
        - DO NOT mention 'Bitovi'.
        Example: 'I found {count} articles about your request.'
        """
    else:
        # --- TU LÓGICA DE LISTADO ORIGINAL (SIN CAMBIOS) ---
        system_prompt = """You are a UI/UX Content Specitalist.
        
        TASK: Format the JSON metadata into high-contrast Markdown cards.
        
        VISUAL HIERARCHY RULES:
        1. TITLE: Use level 2 header '##' and make it a link: ## [TITLE](URL)
        2. METADATA: Use standard text with italics for the details to make them look smaller:
            *Author: AUTHOR* | *Year: YEAR*
        3. SEPARATION: Use a horizontal rule '---' after each item.
        
        STRICT RULES:
        - NO introductory or closing text.
        - NO tables.
        - Title MUST be the largest element.
        - If URL is missing, use '#'.
        - DO NOT mention 'Bitovi'.
        """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Data: {raw_data}")
    ])
    
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state['messages'][-1]
    
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"
    

    else:
        return END