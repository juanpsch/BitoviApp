# Bitovi AI Search & RAG Orchestrator

![Python](https://img.shields.io/badge/python-3.12-blue)
![Framework](https://img.shields.io/badge/orchestration-LangGraph-orange)
![LLM](https://img.shields.io/badge/LLM-Llama_3.1_(Ollama)-white)
![Embeddings](https://img.shields.io/badge/embeddings-nomic--embed--text-lightgrey)
![Frontend](https://img.shields.io/badge/frontend-React_19_+_Vite-61DAFB)
![License](https://img.shields.io/badge/license-MIT-green)

A fully **local, privacy-first RAG agent** that answers technical questions about the
[Bitovi](https://www.bitovi.com/blog) engineering blog. It combines a stateful
**LangGraph** workflow, **hybrid retrieval** (semantic + BM25 + metadata filtering) and
**self-correcting query expansion** to deliver grounded, source-cited answers — with no
external API calls and zero data leaving your machine.

<div align="center">
  <a href="https://www.youtube.com/watch?v=eCbAX8MOTao">
    <img src="https://img.youtube.com/vi/eCbAX8MOTao/maxresdefault.jpg" alt="Demo video" style="width:100%;">
  </a>
  <p><em>▶️ Watch the demo</em></p>
</div>

---

## Table of Contents

- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Data Ingestion Pipeline](#-data-ingestion-pipeline)
- [Observability](#-observability)
- [Roadmap](#-roadmap)

---

## ✨ Key Features

- **Intent-aware routing** — every query is classified as `synthesis`, `listing` or
  `reasoning`, and routed to a specialized generator (conversational answer vs. deterministic
  card list).
- **Hybrid retrieval** — semantic search (MMR) is combined with **BM25Plus** keyword
  re-ranking and strict **metadata filters** (category, year, recency).
- **Self-correction loop** — a quality gate scores retrieved context and automatically
  triggers up to three rounds of **query expansion** when relevance is below threshold.
- **Acronym-aware optimization** — a technical glossary expands terms (RAG, K8s, EDA, …)
  before retrieval to maximize hit rate.
- **Grounded answers with citations** — responses are constrained to retrieved documents
  and surfaced with a dedicated *Sources* section in the UI.
- **100% local & private** — runs entirely on **Ollama** + **ChromaDB**; no third-party
  inference APIs.
- **Full observability** — every node, LLM call and tool invocation is traced via
  **Langfuse**.

---

## 🧠 Architecture

The agent is a directed graph compiled with LangGraph. Each query flows through intent
analysis, strategy selection, query optimization, retrieval, a quality gate, and finally a
specialized generator.

```
            ┌──────────────────┐
            │  intent_analyzer │  synthesis / listing / reasoning
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │     analizar     │  route (Fast | Convencional) + sort_by + top_k
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │  query_optimizer │  expand acronyms via glossary
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │    retrieval     │  forces the retrieve_docs tool
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐      score < threshold (≤ 3 retries)
            │  tools (retrieve)│───────────────┐
            └────────┬─────────┘               ▼
                     │ grade_retrieval   ┌────────────┐
                     │                   │ expansion  │──┐
        score OK ────┤                   └────────────┘  │
                     ▼                                    │ loop back to retrieval
            ┌──────────────────┐                          │
            │ puntos_de_decision│◀────────────────────────┘
            └────────┬─────────┘
          route_generator (by task_type)
             ┌───────┴────────┐
             ▼                ▼
      ┌────────────┐   ┌──────────────────┐
      │ generator  │   │ listing_generator│
      └─────┬──────┘   └────────┬─────────┘
            └──────────┬─────────┘
                       ▼
                      END
```

### Retrieval strategies

| Strategy | When | How |
|----------|------|-----|
| **Fast** | Pure chronology / counts ("latest post", "how many articles") | Direct metadata query against ChromaDB, skips semantic search |
| **Convencional** | Topic / concept / how-to questions | MMR semantic search → BM25Plus re-rank on keywords → dedup by `doc_id` → sort → top-k |

The retrieval tool emits a `LOW_RELEVANCE_ERROR` when the best BM25 score falls below the
threshold, which the `grade_retrieval` gate uses to drive the expansion loop.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph 1.2, LangChain 1.4 |
| LLM | Llama 3.1 8B via **Ollama** |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | ChromaDB (with `ParentDocumentRetriever`) |
| Re-ranking | BM25Plus (`rank-bm25`) |
| Structured outputs | Pydantic v2 |
| API | FastAPI + Uvicorn |
| Frontend | React 19 + Vite + Tailwind CSS 4 |
| Observability | Langfuse |
| Tooling | `uv` (dependency & venv management) |

---

## 📁 Project Structure

```text
.
├── src/
│   ├── agent/                  # Core LangGraph logic
│   │   ├── graph.py            # Workflow compilation, nodes & edges
│   │   ├── nodes.py            # Node implementations (analyzer, optimizer, generators…)
│   │   ├── routers.py          # Conditional routing logic
│   │   └── state.py            # AgentState (TypedDict) definition
│   ├── scripts/                # Data & tools layer
│   │   ├── my_tools.py         # retrieve_docs hybrid retrieval tool
│   │   ├── utils.py            # Retrieval helpers (search, BM25 re-rank, filters)
│   │   ├── schemas.py          # Pydantic models & structured-output schemas
│   │   ├── mapping.py          # Technical glossary & category mappings
│   │   └── mysql_tools.py      # (Standalone) experimental SQL agent — not wired into the graph
│   └── main.py                 # FastAPI entry point (POST /ask)
├── ingest/
│   ├── enrich.py               # LLM-based category enrichment of raw articles
│   ├── indexer2.py             # Builds the Chroma vector DB (parent-document retrieval)
│   └── ingest.ipynb            # Scraping / prototyping notebook
├── app/bitovi-frontend/        # React + Vite + Tailwind UI
├── pyproject.toml              # Python project & dependencies (uv)
├── .env_example                # Environment variable template
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — `pip install uv`
- **[Ollama](https://ollama.com/)** running locally
- **Node.js 18+** (for the frontend)

### 1. Pull the required Ollama models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Configure environment variables

Copy the template and fill in your values:

```bash
cp .env_example .env
```

```ini
LLM_MODEL=llama3.1:8b
BASE_URL=http://localhost:11434
COLLECTION_NAME=bitovi_full_docs
EMBEDDING_MODEL=nomic-embed-text

# Langfuse (optional — for tracing)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Install dependencies

```bash
# Backend (Python)
uv sync

# Frontend
cd app/bitovi-frontend
npm install
```

---

## ▶️ Running the Application

### Backend (FastAPI)

The backend uses absolute imports rooted at `src/`, so launch Uvicorn from that directory:

```bash
cd src
uv run uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` (interactive docs at `/docs`).

### Frontend (React)

```bash
cd app/bitovi-frontend
npm run dev
```

Open the printed URL (default `http://localhost:5173`) and start asking questions.

> **Note:** The vector database must be built before the backend can return results — see
> [Data Ingestion Pipeline](#-data-ingestion-pipeline).

---

## 🔌 API Reference

### `POST /ask`

Submit a natural-language question to the agent.

**Request**

```json
{ "question": "What are the benefits of using Playwright for E2E testing?" }
```

**Response**

```json
{
  "response": "## Markdown answer with citations …",
  "sources": [
    { "title": "Why Playwright", "url": "https://...", "author": "Jane Doe" }
  ]
}
```

On failure the endpoint returns `{ "error": "<message>" }`.

---

## 📚 Data Ingestion Pipeline

The knowledge base is built in two stages from a scraped corpus of blog posts
(`data/bitovi_raw.json`):

1. **Enrich** — `ingest/enrich.py` uses the LLM to tag each article with technical
   categories, producing `data/bitovi_enriched.json`.
2. **Index** — `ingest/indexer2.py` splits articles into child chunks (400 chars), embeds
   them with `nomic-embed-text`, stores them in ChromaDB, and keeps the full parent
   documents in a local store via `ParentDocumentRetriever`. Article dates are normalized
   into `year` and `date_ts` metadata for fast chronological filtering.

```bash
uv run python ingest/enrich.py
uv run python ingest/indexer2.py
```

---

## 📊 Observability

Every request is traced end-to-end with **Langfuse**. When the `LANGFUSE_*` environment
variables are set, each node, LLM call and tool invocation appears in your Langfuse
dashboard under the `api_ask_agent` trace, correlated by `session_id`.

---

## 🗺 Roadmap

- [ ] Multi-turn conversational memory (LangGraph checkpointer)
- [ ] Configurable, corpus-aware relevance threshold (currently a fixed BM25 cutoff)
- [ ] Streaming responses to the frontend
- [ ] Automated evaluation harness (see `src/tests/`)

---

<p align="center"><em>Built with LangGraph · Ollama · ChromaDB · FastAPI · React</em></p>
