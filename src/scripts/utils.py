# # RAG Data Retrieval and Re-Ranking

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from scripts.schemas import ChunkMetadata, RankingKeywords, BlogCategory, SearchControl, RestrievalStragegy
import re
from rank_bm25 import BM25Plus
import os


import dotenv
dotenv.load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

# ChromaDB Configuration (from PageRAG - Data Ingestion)
# configurations

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..","..", "data")
PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
STORE_DIR = os.path.join(DATA_DIR, "parent_store")
CHROMA_DIR = PERSIST_DIR
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
LLM_MODEL = os.getenv('LLM_MODEL')
BASE_URL= os.getenv('BASE_URL')

# ollama pull nomic-embed_text
embeddings= OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL, num_ctx=8192)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, temperature=0)


def define_retrieval_strategy(user_query: str) -> SearchControl:
    system_prompt = """
    You are the Strategic Router for a RAG system. Your goal is to analyze the user's query and decide the most efficient retrieval path.

    RULES:
    # You shall treat the word "Bitovi" as "In the knowledge base". It is not a subject nor a category

### ROUTES:

1. 'Fast' (Metadata-Only):
   - USE CASE: Use this ONLY for generic requests regarding chronology, counts, or recency where NO specific topic, technology, or concept is mentioned.
   - TRIGGER KEYWORDS: "latest", "most recent", "last post", "articles or blogs from 2024", "newest".
   - LOGIC: This is a direct database query. It is used when the answer depends on the date/metadata rather than the content of the text.
   - EXAMPLE: "What was the last article published?" or "Show me the 5 most recent posts." "What is last Bitovi's blog about?

2. 'Convencional' (Semantic + Metadata):
   - USE CASE: Use this whenever the user mentions a specific topic, technology, entity, or "how-to" question. 
   - TRIGGER KEYWORDS: Any tech terms (RAG, AI, DevOps, React), specific names, or conceptual questions ("How to...", "Explain...", "What is...").
   - LOGIC: This uses vector similarity. Even if the user says "latest article about RAG", use Convencional because the system must understand the concept of "RAG" to find the right document.
   - EXAMPLE: "What was the last article published about Kubernetes?" or "Show me the 5 most recent posts about ."

### OUTPUT:
Return a JSON object matching the IntentAnalysis schema with the chosen 'route' and a clear 'justification'.
       
    """
    
    return llm.with_structured_output(RestrievalStragegy).invoke([
        ("system", system_prompt),
        ("user", user_query)
    ])

# ### Extract Filters and Ranking Keywords

def analyze_search_control(user_query: str) -> SearchControl:
    system_prompt = """
    Analiza la consulta del usuario para determinar el CONTROL de la búsqueda.
    
    1. SORTING: Look for keywords such as 'last', 'latest', 'most recent', 'newest', or 'último'. 
     If found, 'sort_by' MUST be set to 'date_ts'. Otherwise, set it to null.

   2. QUANTITY (top_k): 
      - If the user asks to 'list', 'show all', or 'several', set 'top_k' to 10.      
      - If the user asks for a blog, article, post IN SINGULAR, set 'top_k' to 1
      - If the user asks for all blogs, posts or articles, set 'top_k' to 50
      - If the user asks for every blog, post or article, set 'top_k' to 50
      - If ask How many articles, blogs or post, set 'top_k' to 50
      - If the user specifies a number X (e.g., 'give me X', 'last X'), set 'top_k' to that exact value X.
      - Default to 3 if no quantity is specified.

   3. REASONING: Provide a concise explanation of why you chose that quantity and sorting method. 
      Identify which specific part of the user's query led to this decision.
       
    """
    
    return llm.with_structured_output(SearchControl).invoke([
        ("system", system_prompt),
        ("user", user_query)
    ])

def extract_filters(user_query:str):

    llm_structured = llm.with_structured_output(ChunkMetadata)
    # Obtener la lista de categorías permitidas    

    # Preparación de los datos
    current_year = 2026
    # Generamos las reglas: 'Tech1, Tech2' -> 'Category'
    tech_rules = "\n".join([f"- {c.tech_mapping} -> {c.value}" for c in BlogCategory])

    prompt = f"""Extract metadata filters from the query for the technical blog. 
    Return null for fields not mentioned.

    USER QUERY: {user_query}

    YEAR CONTEXT:
    - Current year: {current_year}
    - "Last year" -> {current_year - 1}

    CATEGORY MAPPINGS:
    {tech_rules}

    EXAMPLES:
    "How to use Signals in React?" -> {{"category": "State Management", "year": null}}
    "Mark Repka articles about RAG from 2025" -> {{"category": "RAG", "year": 2025}}
    "Latest post about Kubernetes clusters" -> {{"category": "Kubernetes"}}
    "How to scale an EDA architecture" -> {{"category": "Architecture"}}
    "List me all post related to RAG written in 2024"{{"category": "RAG", "year": 2024}}

    Extract metadata:
    """
    
    metadata = llm_structured.invoke(prompt)
    filters = metadata.model_dump(exclude_none=True)

    return filters


# ### Generate Ranking Keywords

def generate_ranking_keywords(user_query: str):
    # ALT + Z

    # tech_rules = "\n".join([f"{c.tech_mapping} , {c.value}" for c in BlogCategory])  
    tech_rules = "\n".join([f"Keywords: {c.tech_mapping} ===> Category: {c.value}" for c in BlogCategory])

    
    prompt = f"""
            You are a technical classification agent. 
            Your goal is to extract up to 5 technical keywords/categories from the user query based on the ALLOWED MAPPING.

            ALLOWED TECHNOLOGY MAPPING (Format: Keywords -> Category):
            {tech_rules}

            STRICT RULES:
            1. SEMANTIC MATCHING: Match the user query to the most relevant categories even if the casing is different or there are minor typos
            2. RELEVANCE CHECK: If the query is unrelated to the technical stack provided, return an empty list.
            3. NO "BITOVI": Under no circumstances include the word "Bitovi" in the output.
            4. OUTPUT FORMAT: Return EXACTLY 5 strings using the "Category" names from the mapping. If there are fewer than 5 relevant categories, repeat the most relevant ones or add related sub-technologies from the mapping to fill the 5 slots.
            5. CASE INSENSITIVE: Ignore case when searching for matches.

            User Query: "{user_query}"
            Selected Keywords:"""
    
    llm_structured = llm.with_structured_output(RankingKeywords)
    result = llm_structured.invoke(prompt)

    return result.keywords

# ### Search the Doc from Vector DB

def build_search_kwargs(filters, ranking_keywords, k=3):
    # k es el número final, fetch_k es el pool inicial
    search_kwargs = {"k": k, "fetch_k": k * 20}

    # 1. Manejo de Filtros de Metadatos
    if filters:
        filters_conditions = []
        for key, value in filters.items():
            # Mapeo de 'category' a 'categories'
            actual_key = "categories" if key == "category" else key
            
            # Usamos el operador que confirmaste que funciona para tus listas
            if actual_key == "categories":
                # Si es la categoría, forzamos el uso de $contains
                filters_conditions.append({actual_key: {"$contains": value}})
            else:
                # Para otros filtros (como 'year' o 'author'), match directo
                filters_conditions.append({actual_key: value})

        if len(filters_conditions) == 1:
            search_kwargs['filter'] = filters_conditions[0]
        else:
            search_kwargs['filter'] = {"$and": filters_conditions}

    # 2. Manejo de Keywords (Contenido)
    if ranking_keywords:
        if len(ranking_keywords) == 1:
            search_kwargs['where_document'] = {'$contains': ranking_keywords[0]}
        else:
            search_kwargs['where_document'] = {
                "$or": [{'$contains': keyword} for keyword in ranking_keywords]
            }

    return search_kwargs

def get_docs_by_metadata(filters, k=1):
    """
    Recupera los artículos más recientes basándose exclusivamente en filtros 
    de metadatos, saltándose la búsqueda semántica.
    """
    # 1. Consultamos Chroma por Metadata
    # filters ya viene procesado por extract_filters(q)
    # Si filters es {}, Chroma trae todo, lo cual es correcto para "últimos posts"
    raw = vector_store.get(
        where=filters if filters else None,
        include=["metadatas", "documents"]
    )
    
    if not raw or not raw['ids']:
        return []

    # 2. Convertimos a Documentos de LangChain
    docs = [
        Document(page_content=raw['documents'][i], metadata=raw['metadatas'][i])
        for i in range(len(raw['ids']))
    ]
    
    return docs


def search_docs(query, filters={}, ranking_keywords=[], k=3):
    """
        Search documents with metadata and content filters.
        
        Args:
            query (str): Search query text
            filters (dict): Metadata filters (e.g., {"category": "RAG", "year": 2025})
            ranking_keywords (list): Keywords for content filtering (documents must contain at least one)
            k (int): Number of results (default: 5)
        
        Returns:
            list: Matching Document objects
        
        Example:
            docs = search_docs(
                query="List me the last post related to RAG in 2025",
                filters={"category": "RAG", "year": "2025"},
                ranking_keywords=["RAG","Retrieval Augmented Generation","Vector Search","Semantic Search", "MCP"],
                k=10
            )
        """
    search_kwargs = build_search_kwargs(filters, ranking_keywords, k)

    retriever = vector_store.as_retriever(
        search_type= "mmr",
        search_kwargs = search_kwargs
    )

    return retriever.invoke(query)

def process_results(docs, k=3, sort_by=None):
    """
    Aplica de-duplicación, ordenamiento y recorte final de resultados.
    """
    if not docs:
        return []

    # 1. ORDENAMIENTO (Si se solicita)
    if sort_by == "date_ts":
        # Ordenamos por el timestamp que generamos en el indexador
        docs = sorted(
            docs, 
            key=lambda x: x.metadata.get("date_ts", 0), 
            reverse=True
        )

    # 2. DE-DUPLICACIÓN POR DOC_ID
    # Como indexaste con chunks, varios pedazos tienen el mismo doc_id
    unique_docs = []
    seen_ids = set()

    for doc in docs:
        parent_id = doc.metadata.get("doc_id")
        
        if parent_id not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(parent_id)
        
        # 3. TOP K (Recorte final)
        if len(unique_docs) == k:
            break
            
    return unique_docs


def clean_for_bm25(text):
    # En lugar de buscar encabezados, limpiamos el texto para tokenizar
    # Quitamos caracteres especiales pero mantenemos palabras
    content = re.sub(r'[^\w\s]', ' ', text.lower())
    return content.split()

def rank_documents_by_keywords(docs, keywords, k=5):
    """
    Versión robusta: Rankea docs usando BM25Plus sobre el contenido total limpio.
    """
    if not docs or not keywords:
        return docs
    
    # Tokenizamos la query (ej: "react rag" -> ["react", "rag"])
    query_tokens = [kw.lower() for kw in keywords]

    # Preparamos el corpus: una lista de listas de palabras
    # Ya no dependemos de extract_headings_with_content
    corpus = []
    for doc in docs:
        tokens = clean_for_bm25(doc.page_content)
        corpus.append(tokens)

    # Inicializamos BM25 con el corpus procesado
    bm25 = BM25Plus(corpus)
    
    # Obtenemos los scores de relevancia
    doc_scores = bm25.get_scores(query_tokens)

    # Ordenamos los índices de mayor a menor score
    ranked_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)


    # 3. CREAMOS LAS LISTAS SINCRONIZADAS
    # Tomamos solo los mejores 'k' resultados
    final_docs = [docs[i] for i in ranked_indices[:k]]
    final_scores = [doc_scores[i] for i in ranked_indices[:k]]

    # Log de debugging para ver si los scores tienen sentido
    for rank, idx in enumerate(ranked_indices[:k], 1):
        if doc_scores[idx] > 0:
            print(f"   [BM25 Rank {rank}] Doc {idx}: score={doc_scores[idx]:.4f}")

    return {'docs':final_docs, 'scores':final_scores}

