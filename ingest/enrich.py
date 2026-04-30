import os
import json
from langchain_ollama import ChatOllama
import json


import dotenv
dotenv.load_dotenv()



# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
STORE_DIR = os.path.join(DATA_DIR, "parent_store") 
CACHE_FILE = os.path.join(DATA_DIR, "bitovi_raw.json")
ENRICHED_FILE = os.path.join(DATA_DIR, "bitovi_enriched.json")

# CACHE_FILE = "../data/bitovi_raw.json"
# ENRICHED_FILE = "../data/bitovi_enriched.json"



LLM_MODEL = os.getenv('LLM_MODEL')
BASE_URL= os.getenv('BASE_URL')

# Inicializamos el modelo para extracción
llm_extractor = ChatOllama(
    model=LLM_MODEL,
    base_url=BASE_URL,
    temperature=0,
    num_predict=128, # Limitamos la respuesta para que no se extienda y falle
    stop=["\n"]       # Forzamos parada inmediata tras la lista de keywords
)


llm = ChatOllama(model=LLM_MODEL, temperature=0)

CATEGORIES = (
    # Core Technologies
    "React, Angular, Node.js, Next.js, TypeScript, "
    # AI & Data
    "AI & Machine Learning, LLMs, RAG, Data Science, "
    # DevOps & Cloud
    "DevOps, Kubernetes, AWS, Terraform, Infrastructure as Code, "
    # Testing & Quality
    "E2E Testing, Playwright, Cypress, Unit Testing, "
    # Specialized Frontend
    "State Management, Web Components, Performance, Module Federation, "
    # Process & Business
    "Open Source, Project Management, Product Design, UX/UI, Agile"
)

def enrich_process():
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total a procesar: {len(data)}")
    
    enriched_data = []
    
    for i, item in enumerate(data):
        # Si el item ya tiene categorías (por si reintentas), te lo saltás
        if "categories" in item["metadata"]:
            enriched_data.append(item)
            continue
            
        prompt = f"""Analyze this technical snippet and classify it into 3 to 5 keywords.
        Choose primarily from: [{CATEGORIES}].
        Return ONLY keywords separated by commas.
        Text: {item['page_content'][:1000]}"""
        
        try:
            res = llm.invoke(prompt)
            item["metadata"]["categories"] = res.content.strip().replace("\n", " ")
            print(f"[{i+1}/{len(data)}] Etiquetado: {item['metadata']['title'][:30]}...")
        except Exception as e:
            print(f"Error en {i}: {e}")
            item["metadata"]["categories"] = "General"
            
        enriched_data.append(item)
        
        # Guardado de seguridad cada 10 docs
        if (i + 1) % 10 == 0:
            with open(ENRICHED_FILE, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=4)
                
    # Guardado final
    with open(ENRICHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=4)
    print("✅ Enriquecimiento terminado.")

if __name__ == "__main__":
    enrich_process()


