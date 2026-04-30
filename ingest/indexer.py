import os
import shutil
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import dotenv
dotenv.load_dotenv()



# Obtiene la ruta de la carpeta donde está parado este script (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sube un nivel y entra a data
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
CACHE_FILE = os.path.join(DATA_DIR, "bitovi_raw.json")

print(f"Ruta calculada: {PERSIST_DIR}")

def load_raw_data(filepath):
    """Carga los artículos desde el archivo JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el caché en {filepath}. Corré el scraper primero.")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]

def build_vector_db(chunk_size=1000, overlap=150):
    """Limpia la DB anterior y crea una nueva con los parámetros deseados."""
    
    persist_dir = PERSIST_DIR
    cache_file = CACHE_FILE

    # 1. LIMPIEZA: Borrar la DB anterior para evitar duplicados
    if os.path.exists(persist_dir):
        print(f"Borrando base de datos vieja en {persist_dir}...")
        shutil.rmtree(persist_dir)

    # 2. CARGA: Desde el JSON local
    print(f"Cargando datos desde {cache_file}...")
    raw_docs = load_raw_data(cache_file)

    # 3. SPLIT: Aquí es donde jugás con los parámetros
    print(f"Dividiendo {len(raw_docs)} artículos (Size: {chunk_size}, Overlap: {overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=overlap
    )
    docs = text_splitter.split_documents(raw_docs)

    # 4. INDEXADO: Generación de embeddings y persistencia
    print(f"Generando embeddings para {len(docs)} fragmentos... (esto puede tardar)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print("✅ ¡Base de datos Chroma actualizada y lista para usar!")
    return vectorstore

if __name__ == "__main__":
    # Probá diferentes valores aquí sin miedo a romper nada
    build_vector_db(chunk_size=1200, overlap=200)