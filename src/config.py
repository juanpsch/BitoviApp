import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434")


llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, temperature=0)