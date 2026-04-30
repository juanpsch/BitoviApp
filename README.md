# Bitovi AI Search & RAG Orchestrator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/framework-LangGraph-orange)
![LLM](https://img.shields.io/badge/LLM-Gemini_1.5_Flash-green)

## 🚀 Overview
This repository contains a production-grade **AI Agent** designed to navigate and synthesize Bitovi's technical content. Built with **LangGraph**, the system implements a state-of-the-art **Retrieval-Augmented Generation (RAG)** pipeline that transforms informal user inquiries into precise, technically-grounded insights.

The agent is engineered to handle complex technical queries, resolving acronyms through a custom glossary and maintaining strict "grounding" to ensure every response is backed by internal Bitovi documentation.

## 🛠 Technical Stack
*   **Orchestration**: LangGraph (Stateful Graph) & LangChain.
*   **State Management**: Pydantic for strict schema validation.
*   **Retrieval**: Hybrid Search (BM25 + Semantic) with threshold-based retry logic.
*   **Acronym Resolution**: Custom internal glossary for technical mapping (K8s, RAG, CI/CD, etc.).
*   **Backend**: Python 3.10+.

## 🧠 System Architecture
The agent operates as a directed acyclic graph (DAG) to ensure high-quality output:

1.  **Intent Analyzer**: Categorizes queries into `SINTESIS`, `LISTING`, or `REASONING`.
2.  **Query Optimizer**: Resolves acronyms using a technical glossary and enforces corporate exclusion rules (e.g., stripping "Bitovi" from internal search strings to reduce noise).
3.  **Expansion Node**: Automatically triggers if the retrieval score is below **0.6**, expanding the query with technical keywords (PyTorch, TensorFlow, etc.) to improve hit rates.
4.  **Generator Node**: A grounded LLM node that synthesizes the final answer using **only** provided documents, preventing hallucinations and external knowledge leaks.

## 📋 Key Features
*   **Acronym Expansion**: Automatically translates terms like "IA" to "Artificial Intelligence" to maximize search relevance.
*   **Strict Grounding**: The system is hard-coded to refuse answering from external training data, citing only provided URLs from the Bitovi blog.
*   **Metadata Filtering**: Optimized for technical keywords, stripping common stopwords and conversational noise.
*   **Retry Logic**: Autonomous query expansion if initial results are of low relevance.

## ⚙️ Installation

1.  **Clone the Repo**:
    ```bash
    git clone [https://github.com/bitovi/ai-rag-orchestrator.git](https://github.com/bitovi/ai-rag-orchestrator.git)
    cd ai-rag-orchestrator
    ```

2.  **Set Up Environment**:
    Create a `.env` file:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    VECTOR_DB_ENDPOINT=your_endpoint
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Execute the graph via the LangGraph `CompiledGraph` interface:
```python
from src.graph import app

inputs = {"messages": [("user", "What is RAG?")]}
config = {"configurable": {"thread_id": "bitovi_session_01"}}

for output in app.stream(inputs, config):
    # The agent manages state transitions between Optimizer, Retriever, and Generator
    print(output)
