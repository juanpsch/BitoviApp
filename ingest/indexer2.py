import os
import shutil
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- NUEVOS IMPORTS PARA PARENT RETRIEVAL ---
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore

import dotenv
dotenv.load_dotenv()



# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
STORE_DIR = os.path.join(DATA_DIR, "parent_store") # Nueva carpeta para los docs completos
CACHE_FILE = os.path.join(DATA_DIR, "bitovi_enriched.json")

from datetime import datetime

def load_raw_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el caché en {filepath}.")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    for d in data:
        metadata = d["metadata"]
        date_str = metadata.get("date", "")
        
        if date_str:
            try:
                # 1. Convertimos a objeto datetime
                dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
                
                # 2. Guardamos el AÑO como INT (para filtros rápidos)
                metadata["year"] = dt_obj.year
                
                # 3. Guardamos la FECHA como TIMESTAMP (INT)
                # Esto permite hacer: filter={"date": {"$gte": 1745337600}}
                metadata["date_ts"] = int(dt_obj.timestamp())
                
                # Opcional: Mantener el string original para lectura humana
                # metadata["date"] = date_str 
                
            except ValueError:
                print(f"⚠️ Error de formato en fecha: {date_str}")
                metadata["year"] = 0
                metadata["date_ts"] = 0
        
        docs.append(Document(page_content=d["page_content"], metadata=metadata))
    
    return docs

def build_vector_db():
    # 1. LIMPIEZA
    for path in [PERSIST_DIR, STORE_DIR]:
        if os.path.exists(path):
            print(f"Borrando {path}...")
            shutil.rmtree(path)
    os.makedirs(STORE_DIR, exist_ok=True)

    # 2. CARGA
    raw_docs = load_raw_data(CACHE_FILE)

    # 3. CONFIGURACIÓN DE SPLITTERS
    # Chunks pequeños para que la búsqueda semántica sea muy precisa
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # IMPORTANTE: No usamos parent_splitter si queremos el documento COMPLETO.
    # Al pasar None, LangChain guarda el documento original entero en el store.

    # 4. COMPONENTES DE PERSISTENCIA
    # Configuramos BGE-M3 a través de Ollama
    embeddings = OllamaEmbeddings(
    model="nomic-embed-text",    
    base_url="http://localhost:11434" 
)
    
    # El vectorstore para los pedacitos (Child)
    vectorstore = Chroma(
        collection_name="bitovi_full_docs",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    # El almacenamiento para los documentos enteros (Parent)
    fs = LocalFileStore(STORE_DIR)
    store = create_kv_docstore(fs)

    # 5. EL RETRIEVER QUE COORDINA TODO
    print(f"Indexando {len(raw_docs)} documentos completos...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        # parent_splitter=None  <-- Si no lo pones, guarda el doc original
    )

    # 6. AGREGAR DOCUMENTOS
    # Esto genera los chunks, los mete en Chroma y guarda el original en STORE_DIR       

    print("✅ ¡Indexación completada! Ahora recuperás documentos enteros.")
    # Definimos un tamaño de lote pequeño (ej. 50 documentos originales a la vez)
    # para no saturar los parámetros de SQLite
    batch_size = 50
    for i in range(0, len(raw_docs), batch_size):
        batch = raw_docs[i : i + batch_size]
        print(f" -> Procesando lote {i//batch_size + 1} ({len(batch)} documentos)...")
        retriever.add_documents(batch, ids=None)
    return retriever

if __name__ == "__main__":
    build_vector_db()