# Design: Live agent-flow streaming in the frontend

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Branch:** `fix/rag-agent-improvements`

## Goal

Show the agent's reasoning flow in the frontend **in real time** while it answers:
a side panel with the sequence of steps the agent takes (friendly phases + a
collapsible technical detail per step), plus the final answer streamed **token by
token** in the main area.

## Decisions (from brainstorming)

| Topic | Decision |
|-------|----------|
| Detail level | **Hybrid (C):** friendly phase labels always visible + collapsible "technical detail" per step |
| Final answer | **Streamed token by token (B)** (ChatGPT-style) |
| Steps after completion | **Stay visible**, in a **side panel** (left column) |
| Streaming mechanism | LangGraph `astream(stream_mode=["updates","messages"])` over **SSE** |

## Architecture

Single graph run streamed over Server-Sent Events. The frontend consumes the SSE
stream via `fetch` + `ReadableStream` (POST rules out `EventSource`).

```
React (fetch + ReadableStream reader)
        │  POST /ask/stream   (text/event-stream)
        ▼
FastAPI StreamingResponse
        │  async for mode, chunk in agent_graph.astream(
        │       inputs, config, stream_mode=["updates","messages"])
        ▼
LangGraph agent  ──►  Ollama (llama3.1:8b, nomic-embed-text) + ChromaDB
```

- `updates` mode → per-node state updates → emitted as `step` events.
- `messages` mode (filtered to `generator`/`listing_generator`) → emitted as `token` events.
- After the loop → `sources` event (parsed from the last successful ToolMessage) + `done`.

## Backend

### Endpoint

`POST /ask/stream` in `src/main.py`, returning
`StreamingResponse(event_stream(question), media_type="text/event-stream")`.

The existing `POST /ask` (non-streaming) is **kept** for programmatic use. Both share
helpers (`_extract_sources`, config builder).

### SSE protocol

Each message is a single SSE `data:` line followed by a blank line, carrying JSON with
a `type` discriminator:

```
data: {"type":"step","id":"intent_analyzer","label":"🔍 Entendiendo tu pregunta","detail":{...}}\n\n
```

| `type` | Payload | When |
|--------|---------|------|
| `step` | `{id, label, detail}` | a relevant node completes |
| `token` | `{text}` | each LLM answer fragment |
| `sources` | `{sources: [...]}` | end, parsed from last ToolMessage |
| `done` | `{}` | stream finished |
| `error` | `{message}` | exception during the run |

### Node → step mapping (`build_step(node_name, update) -> dict | None`)

Pure function. Returns `None` for omitted nodes (`retrieval`, `puntos_de_decision`) to
avoid noise.

| Node | Label | Technical detail |
|------|-------|------------------|
| `intent_analyzer` | 🔍 Entendiendo tu pregunta | `task_type`, `intent_reasoning` |
| `analizar` | 🧭 Eligiendo estrategia | `selected_route`, `sort_by`, `top_k` |
| `query_optimizer` | ✨ Optimizando la búsqueda | `search_query` |
| `tools` | 📚 Buscando en el blog | strategy, doc count, BM25 scores (parsed from ToolMessage JSON) |
| `expansion` | 🔁 Refinando la búsqueda (intento N) | new `search_query`, `retry_count` |
| `generator` | ✍️ Redactando respuesta | — (tokens stream separately) |
| `listing_generator` | 🗂️ Armando el listado | — |

### Token streaming & no-token cases

`stream_mode="messages"` yields `(message_chunk, metadata)`; emit a `token` only when
`metadata["langgraph_node"] in {"generator","listing_generator"}`.

`listing_generator` is deterministic (no LLM) and the "no documents" fallback builds the
`AIMessage` without an LLM call — neither streams tokens. The generator tracks a
`streamed_any_token` flag; when the `updates` event for a generator-type node arrives and
no token was streamed, it emits the message's full `content` as a single `token`. This
guarantees the frontend always receives the answer the same way.

### Sources

During the `updates` stream, capture the latest valid `tools` ToolMessage content
(skip `LOW_RELEVANCE_ERROR` / `No documents found`). At the end, run `_extract_sources`
on it and emit the `sources` event.

### Errors & observability

- The whole `astream` loop is wrapped in try/except; on exception emit
  `{"type":"error","message":...}` and close the stream cleanly.
- The existing UTF-8 `stdout`/`stderr` reconfigure stays (node `print()`s still land in
  `backend.log`).
- Langfuse `CallbackHandler` is passed in the config exactly as in `/ask`.

## Frontend

Two-column responsive layout in `app/bitovi-frontend/src/App.jsx` (on mobile the panel
stacks above the answer).

```
┌───────────────────────────────────────────────────────┐
│                 Bitovi AI Expert                       │
│        [ input pregunta ........... ] [Consultar]      │
├──────────────────────┬────────────────────────────────┤
│  PANEL LATERAL        │   RESPUESTA (área principal)    │
│  (Flujo del agente)   │   ## What is RAG? ▌(streaming)  │
│  🔍 Entendiendo…   ✓  │   ...                           │
│  🧭 Estrategia     ✓  │   ── Fuentes citadas ──         │
│     └ ver detalle ▸   │   [card] [card]                 │
│  ✍️ Redactando    ⟳   │                                 │
└──────────────────────┴────────────────────────────────┘
```

### State

`steps[]` (`{id, label, detail, status}`), `answer` (growing string), `sources[]`,
`loading`.

### Stream reading

```js
const res = await fetch('http://localhost:8000/ask/stream', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question }),
})
const reader = res.body.getReader()
const decoder = new TextDecoder()
let buffer = ''
while (true) {
  const { done, value } = await reader.read()
  if (done) break
  buffer += decoder.decode(value, { stream: true })
  const parts = buffer.split('\n\n')
  buffer = parts.pop()
  for (const part of parts) handleEvent(JSON.parse(part.replace(/^data: /, '')))
}
```

### `handleEvent(evt)`

- `step` → append to `steps` with a ✓ (events arrive post-completion); a pulsing
  "⟳ en curso…" placeholder sits at the panel's bottom while `loading`.
- `token` → `answer += evt.text` (live re-render through the existing `ReactMarkdown`).
- `sources` → `setSources`.
- `done` → `setLoading(false)`.
- `error` → show message, stop spinner.

Each step with a `detail` shows a `▸ ver detalle` toggle that expands the raw data
(route, top_k, keywords, scores) — the "hybrid C" requirement.

Reuses existing Tailwind styling and the `ReactMarkdown` component overrides.

## Testing

Pure functions are the testable seam (no Ollama needed):

- **Unit** (`src/tests/test_streaming.py`, pytest):
  - `build_step` maps each node to the right label/detail and returns `None` for omitted nodes.
  - SSE formatting produces parseable JSON.
  - `_extract_sources` over a sample ToolMessage.
- **Manual integration:** `curl -N -X POST .../ask/stream` to watch live events; browser
  test with a synthesis query (tokens) and a listing query (no tokens).

## Scope

**In:**
| File | Change |
|------|--------|
| `src/main.py` | `+ /ask/stream`, `+ build_step`, SSE formatting; refactor shared helpers |
| `app/bitovi-frontend/src/App.jsx` | 2-column layout, stream reading, side panel, token render |
| `src/tests/test_streaming.py` | new — unit tests for pure functions |

**Out (YAGNI):** no changes to node logic, no memory/persistence, no streaming of
intermediate LLM calls (only the final generator), no auth.
