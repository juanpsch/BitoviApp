
# Bitovi AI Search & RAG Orchestrator

![Python](https://img.shields.io/badge/python-3.12-blue)
![Framework](https://img.shields.io/badge/framework-LangGraph-orange)
![LLM](https://img.shields.io/badge/LLM-Llama_3.1_(Ollama)-white)
![Embeddings](https://img.shields.io/badge/Embeddings-Nomic--Text-lightgrey)

## 🚀 Overview
An advanced **Autonomous RAG Agent** engineered for Bitovi's technical ecosystem. This system utilizes **LangGraph** to manage complex state transitions, ensuring high-fidelity knowledge retrieval and synthesis from local LLM instances.

<div align="center">
  <a href="https://www.youtube.com/watch?v=eCbAX8MOTao">
    <img src="https://img.youtube.com/vi/eCbAX8MOTao/maxresdefault.jpg" alt="Ver video" style="width:100%;">
  </a>
</div>

## 🛠 Tech Stack
*   **Engine**: Python 3.12
*   **LLM**: Llama 3.1 (via Ollama)
*   **Embeddings**: `nomic-embed-text`
*   **Orchestration**: LangGraph & LangChain
*   **State Management**: Pydantic-validated `AgentState`




## 🧠 Graph Architecture
The agent follows a sophisticated conditional workflow to ensure quality:

1.  **Intent Analyzer**: Categorizes query intent.
2.  **Get query parameters**: Obtain from the query sorting stategy and top_k elements.
3.  **Retrieval Strategy**: Set the best retrieval strategy for the requested data set.
4.  **Query Optimizer**: Resolves acronyms (e.g., K8s, RAG) using a custom technical glossary.
5.  **Retrieval Node**: Interacts with the `retrieve_docs` tool.
6.  **Grade Retrieval**: A self-correction layer that triggers the **Expansion Node** if the context relevance score is below threshold ($< 1.0$).
7.  **Dynamic Routing**: A logical gateway routes the state to specialized generators (`Standard Generator` vs. `Listing Generator`) based on the task type.


The following section details the logical flow of the system built with LangGraph. The agent performs intent analysis, optimizes vector database retrieval, and dynamically determines whether additional tools or query expansion are required before generating the final response.
### Graph Structure
<img width="391" height="853" alt="graph" src="https://github.com/user-attachments/assets/7a537798-bfb9-4642-a1a1-891539ed7a02" />


## 📁 Repository Structure
```text
src/
.
├── agent/                  # Core graph logic
│   ├── graph.py            # Workflow compilation & edges
│   ├── nodes.py            # Node definitions (Optimizer, Analyzer, etc.)
│   ├── routers.py          # Conditional routing logic
│   ├── state.py            # Pydantic AgentState definition
│   └── __init__.py
├── debug_logs/             # RAG evaluation & trace logs
│   └── retrieved_reranked_docs.md
├── juptyer_tests/          # Prototyping & ChromaDB experiments
│   ├── chroma.ipynb
│   └── nuevo.ipynb
├── scripts/                # Data & Tools layer
│   ├── mapping.py          # Glossary & Technical mappings
│   ├── mysql_tools.py      # Database persistence
│   ├── my_tools.py         # Retrieval tools (retrieve_docs)
│   ├── schemas.py          # Pydantic models & structured outputs
│   ├── utils.py            # Helper functions
│   └── __init__.py
├── .env                    # Environment variables (excluded from Git)
├── config.py               # Global configuration
├── main.py                 # Application entry point
└── requirements.txt        # Project dependencies

