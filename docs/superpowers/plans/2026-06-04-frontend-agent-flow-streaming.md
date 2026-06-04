# Live Agent-Flow Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the agent's reasoning steps in a live side panel while the final answer streams token by token in the main area.

**Architecture:** A new `POST /ask/stream` FastAPI endpoint runs the LangGraph agent once with `astream(stream_mode=["updates","messages"])` and emits Server-Sent Events (`step`, `token`, `sources`, `done`, `error`). The React frontend reads the SSE stream with `fetch` + `ReadableStream` and renders a two-column layout: a step timeline on the left and the streaming answer on the right.

**Tech Stack:** FastAPI `StreamingResponse`, LangGraph `astream`, React 19 `fetch`/`ReadableStream`, Tailwind CSS, pytest.

**Spec:** `docs/superpowers/specs/2026-06-04-frontend-agent-flow-streaming-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Add `pytest` dev dependency + pytest `pythonpath=["src"]` config |
| `src/main.py` | Add pure helpers (`_val`, `_format_sse`, `_sources_from_tool_content`, `build_step`, `_tool_step_detail`), refactor `_extract_sources` + `_build_config`, add `POST /ask/stream` |
| `src/tests/test_streaming.py` | Unit tests for the pure helpers |
| `app/bitovi-frontend/src/App.jsx` | Two-column layout, SSE stream reading, step panel, token rendering |

**Note on design refinement:** The spec listed `generator`/`listing_generator` in the `build_step` table. For correct event ordering (the "writing" step must appear *with* the first token, not after the node finishes), those two nodes are handled inside the streaming loop instead — `build_step` returns `None` for them. All other behavior matches the spec.

---

## Task 1: Add pytest dev dependency and config

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pytest as a dev dependency**

Run:
```bash
uv add --dev pytest
```
Expected: `pyproject.toml` gains a `[dependency-groups]` (or `[tool.uv]` dev) entry with `pytest`, and `uv.lock` updates.

- [ ] **Step 2: Configure pytest to put `src` on the path**

Add this block to the end of `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["src/tests"]
```

- [ ] **Step 3: Verify pytest runs (collects 0 tests, no errors)**

Run:
```bash
uv run pytest -q
```
Expected: exits 0 (or "no tests ran"); no import/config errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pytest dev dependency and config"
```

---

## Task 2: Backend pure helpers (TDD)

These are pure, side-effect-free functions added to `src/main.py`. We write the tests first.

**Files:**
- Create: `src/tests/test_streaming.py`
- Modify: `src/main.py`

- [ ] **Step 1: Write the failing tests**

Create `src/tests/test_streaming.py`:
```python
import json

from langchain_core.messages import ToolMessage

from main import build_step, _format_sse, _sources_from_tool_content


def test_format_sse_is_parseable():
    out = _format_sse({"type": "token", "text": "hola"})
    assert out.startswith("data: ")
    assert out.endswith("\n\n")
    payload = json.loads(out[len("data: "):].strip())
    assert payload == {"type": "token", "text": "hola"}


def test_format_sse_keeps_unicode_readable():
    out = _format_sse({"type": "step", "label": "🔍 Entendiendo"})
    assert "🔍" in out  # ensure_ascii=False


def test_build_step_intent_analyzer():
    step = build_step(
        "intent_analyzer",
        {"task_type": "listing", "intent_reasoning": "pide una lista"},
    )
    assert step["type"] == "step"
    assert step["id"] == "intent_analyzer"
    assert "Entendiendo" in step["label"]
    assert step["detail"]["razonamiento"] == "pide una lista"
    assert step["detail"]["intencion"] == "listing"


def test_build_step_analizar_reads_route_and_params():
    step = build_step(
        "analizar",
        {"selected_route": "Convencional", "search_params": {"sort_by": None, "top_k": 10}},
    )
    assert "estrategia" in step["label"].lower() or "Eligiendo" in step["label"]
    assert step["detail"]["ruta"] == "Convencional"
    assert step["detail"]["top_k"] == 10


def test_build_step_tools_reads_metadata_log():
    content = json.dumps(
        {"metadata_log": {"strategy": "CONVENCIONAL", "sort_by": None, "count": 3}, "documents": []}
    )
    step = build_step("tools", {"messages": [ToolMessage(content=content, tool_call_id="x")]})
    assert step["label"].startswith("📚")
    assert step["detail"]["documentos"] == 3
    assert step["detail"]["estrategia"] == "CONVENCIONAL"


def test_build_step_expansion_includes_attempt():
    step = build_step("expansion", {"search_query": "rag vector", "retry_count": 2})
    assert "2" in step["label"]
    assert step["detail"]["query"] == "rag vector"


def test_build_step_returns_none_for_internal_and_generator_nodes():
    assert build_step("retrieval", {"messages": []}) is None
    assert build_step("puntos_de_decision", {}) is None
    # generation steps are emitted by the streaming loop, not build_step
    assert build_step("generator", {"messages": []}) is None
    assert build_step("listing_generator", {"messages": []}) is None


def test_sources_from_tool_content_dedups_and_skips_invalid():
    content = json.dumps(
        {
            "documents": [
                {"title": "T1", "author": "A1", "url": "https://x/1"},
                {"title": "T2", "author": "A2", "url": "https://x/1"},  # duplicate url
                {"title": "T3", "author": "A3", "url": "No URL"},        # skipped
            ]
        }
    )
    sources = _sources_from_tool_content(content)
    assert sources == [{"title": "T1", "url": "https://x/1", "author": "A1"}]


def test_sources_from_tool_content_ignores_error_payloads():
    assert _sources_from_tool_content("LOW_RELEVANCE_ERROR: score 0.2") == []
    assert _sources_from_tool_content("No documents found for the query: 'x'.") == []
    assert _sources_from_tool_content("not json") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
uv run pytest src/tests/test_streaming.py -q
```
Expected: FAIL — `ImportError: cannot import name 'build_step' from 'main'` (helpers not defined yet).

- [ ] **Step 3: Implement the helpers in `src/main.py`**

In `src/main.py`, add these functions just above the existing `_extract_sources` definition:
```python
def _val(value):
    """Return the .value of an Enum, otherwise the value unchanged."""
    return getattr(value, "value", value)


def _format_sse(payload: dict) -> str:
    """Serialize a payload as a single Server-Sent Event message."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sources_from_tool_content(content) -> list:
    """Parse the sources list from a single retrieve_docs ToolMessage payload."""
    if not isinstance(content, str):
        return []
    if content.startswith("LOW_RELEVANCE_ERROR") or content.startswith("No documents found"):
        return []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    documents = data.get("documents") if isinstance(data, dict) else None
    if not documents:
        return []

    sources, seen = [], set()
    for doc in documents:
        url = doc.get("url")
        if not url or url in {"No URL", "#"} or url in seen:
            continue
        sources.append({
            "title": doc.get("title") or "Bitovi Blog Post",
            "url": url,
            "author": doc.get("author") or "Bitovi Expert",
        })
        seen.add(url)
    return sources


def _tool_step_detail(update: dict):
    """Build the technical detail for the 'tools' step from its ToolMessage."""
    messages = update.get("messages", [])
    if not messages:
        return None
    content = messages[-1].content
    if isinstance(content, str) and content.startswith("LOW_RELEVANCE_ERROR"):
        return {"estado": "relevancia baja, se expandirá la query"}
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    log = data.get("metadata_log", {}) if isinstance(data, dict) else {}
    return {
        "estrategia": log.get("strategy"),
        "documentos": log.get("count"),
        "orden": log.get("sort_by"),
    }


def build_step(node_name: str, update: dict):
    """Map a completed graph node + its state update to a friendly UI step.

    Returns None for internal/router nodes and for the generator nodes
    (whose steps are emitted by the streaming loop alongside the first token).
    """
    if node_name == "intent_analyzer":
        return {
            "type": "step",
            "id": node_name,
            "label": "🔍 Entendiendo tu pregunta",
            "detail": {
                "intencion": _val(update.get("task_type")),
                "razonamiento": update.get("intent_reasoning"),
            },
        }
    if node_name == "analizar":
        params = update.get("search_params", {}) or {}
        return {
            "type": "step",
            "id": node_name,
            "label": "🧭 Eligiendo estrategia",
            "detail": {
                "ruta": _val(update.get("selected_route")),
                "sort_by": params.get("sort_by"),
                "top_k": params.get("top_k"),
            },
        }
    if node_name == "query_optimizer":
        return {
            "type": "step",
            "id": node_name,
            "label": "✨ Optimizando la búsqueda",
            "detail": {"query": update.get("search_query")},
        }
    if node_name == "tools":
        return {
            "type": "step",
            "id": node_name,
            "label": "📚 Buscando en el blog",
            "detail": _tool_step_detail(update),
        }
    if node_name == "expansion":
        retries = update.get("retry_count", 0)
        return {
            "type": "step",
            "id": node_name,
            "label": f"🔁 Refinando la búsqueda (intento {retries})",
            "detail": {"query": update.get("search_query")},
        }
    return None
```

Also confirm `json` is already imported at the top of `src/main.py` (it is). No new imports needed for this task.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
uv run pytest src/tests/test_streaming.py -q
```
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/main.py src/tests/test_streaming.py
git commit -m "feat: add SSE/step/sources pure helpers for streaming"
```

---

## Task 3: Add the `/ask/stream` endpoint

Refactor the shared config builder, reuse `_sources_from_tool_content` inside `_extract_sources`, and add the streaming endpoint.

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Add the `StreamingResponse` import**

In `src/main.py`, change the FastAPI import line:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
```

- [ ] **Step 2: Refactor `_extract_sources` to reuse the pure helper**

Replace the existing `_extract_sources` body with:
```python
def _extract_sources(messages) -> list:
    """Rebuild the sources list from the last successful retrieve_docs ToolMessage."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            sources = _sources_from_tool_content(msg.content)
            if sources:
                return sources
    return []
```

- [ ] **Step 3: Add a shared config builder**

Add above the `@app.post("/ask")` route:
```python
def _build_config() -> dict:
    """Build the per-request LangGraph config (Langfuse handler + session id)."""
    session_id = str(uuid.uuid4())
    return {
        "configurable": {},
        "recursion_limit": 20,
        "callbacks": [CallbackHandler()],
        "metadata": {
            "session_id": session_id,
            "langfuse_trace_name": "api_ask_agent",
        },
    }
```

- [ ] **Step 4: Use `_build_config` in the existing `/ask` route**

In `ask_agent`, replace the inline `langfuse_handler = CallbackHandler()` + `config = {...}` block with:
```python
        config = _build_config()
```
(Delete the now-unused `langfuse_handler` line and the inline `config` dict. Keep the rest of `ask_agent` unchanged.)

- [ ] **Step 5: Add the streaming endpoint**

Add this route immediately after the `ask_agent` function:
```python
_GEN_LABELS = {
    "generator": "✍️ Redactando respuesta",
    "listing_generator": "🗂️ Armando el listado",
}


@app.post("/ask/stream")
async def ask_agent_stream(payload: AskRequest):
    config = _build_config()
    inputs = {"messages": [HumanMessage(content=payload.question)]}

    async def event_stream():
        streamed_token = False
        gen_step_sent = False
        last_tool_content = None
        try:
            async for mode, chunk in agent_graph.astream(
                inputs, config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node_name, update in chunk.items():
                        if node_name == "tools":
                            msgs = update.get("messages", [])
                            if msgs:
                                last_tool_content = msgs[-1].content

                        step = build_step(node_name, update)
                        if step:
                            yield _format_sse(step)

                        # No-token generators (listing / "no documents" fallback):
                        # emit their step + full content here.
                        if node_name in _GEN_LABELS and not streamed_token:
                            if not gen_step_sent:
                                yield _format_sse({
                                    "type": "step", "id": node_name,
                                    "label": _GEN_LABELS[node_name], "detail": None,
                                })
                                gen_step_sent = True
                            msgs = update.get("messages", [])
                            text = getattr(msgs[-1], "content", "") if msgs else ""
                            if text:
                                yield _format_sse({"type": "token", "text": text})

                elif mode == "messages":
                    msg_chunk, metadata = chunk
                    node = metadata.get("langgraph_node")
                    if node in _GEN_LABELS:
                        if not gen_step_sent:
                            yield _format_sse({
                                "type": "step", "id": node,
                                "label": _GEN_LABELS[node], "detail": None,
                            })
                            gen_step_sent = True
                        text = getattr(msg_chunk, "content", "")
                        if text:
                            streamed_token = True
                            yield _format_sse({"type": "token", "text": text})

            sources = _sources_from_tool_content(last_tool_content) if last_tool_content else []
            yield _format_sse({"type": "sources", "sources": sources})
            yield _format_sse({"type": "done"})
        except Exception as e:
            print(f"--- [STREAM ERROR] {e} ---")
            yield _format_sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 6: Verify the app still imports**

Run:
```bash
cd src && python -c "import main; print(sorted(r.path for r in main.app.routes if getattr(r,'methods',None)))" && cd ..
```
Expected: includes both `'/ask'` and `'/ask/stream'`.

- [ ] **Step 7: Restart the backend and verify the stream end-to-end (requires Ollama)**

Restart the running backend (stop the background task, relaunch as before), then:
```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream -H "Content-Type: application/json" -d '{"question":"What is RAG?"}'
```
Expected: a sequence of `data: {"type":"step",...}` lines (intent → estrategia → optimizando → buscando → redactando), then many `data: {"type":"token",...}` lines, then `data: {"type":"sources",...}` and `data: {"type":"done"}`.

If NO `token` events appear (only one big token at the end), token streaming from `ChatOllama.invoke` under `stream_mode="messages"` is not firing; the no-token fallback still delivers the full answer, so the feature degrades gracefully. Note it but do not block.

- [ ] **Step 8: Commit**

```bash
git add src/main.py
git commit -m "feat: add POST /ask/stream SSE endpoint for live agent flow"
```

---

## Task 4: Frontend two-column streaming UI

Replace `App.jsx` with a two-column layout that reads the SSE stream.

**Files:**
- Modify: `app/bitovi-frontend/src/App.jsx`

- [ ] **Step 1: Replace `App.jsx` with the streaming UI**

Overwrite `app/bitovi-frontend/src/App.jsx` with:
```jsx
import ReactMarkdown from 'react-markdown'
import { useState } from 'react'

const markdownComponents = {
  a: ({ node, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer"
      className="text-blue-600 font-bold hover:text-blue-800 underline decoration-blue-300 underline-offset-4" />
  ),
  li: ({ node, ...props }) => <li {...props} className="mb-2 ml-4 list-disc" />,
  h2: ({ node, ...props }) => (
    <h2 {...props} className="text-xl font-bold text-blue-900 mt-6 mb-4 border-b pb-2" />
  ),
}

function StepDetail({ detail }) {
  const [open, setOpen] = useState(false)
  if (!detail) return null
  const entries = Object.entries(detail).filter(([, v]) => v !== null && v !== undefined && v !== '')
  if (entries.length === 0) return null
  return (
    <div className="mt-1">
      <button onClick={() => setOpen((o) => !o)}
        className="text-[11px] text-slate-400 hover:text-slate-600">
        {open ? '▾ ocultar detalle' : '▸ ver detalle'}
      </button>
      {open && (
        <dl className="mt-1 text-[11px] text-slate-500 bg-slate-50 rounded-lg p-2 space-y-0.5">
          {entries.map(([k, v]) => (
            <div key={k} className="flex gap-2">
              <dt className="font-semibold">{k}:</dt>
              <dd className="break-all">{String(v)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

function App() {
  const [question, setQuestion] = useState('')
  const [steps, setSteps] = useState([])
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)

  const handleEvent = (evt) => {
    switch (evt.type) {
      case 'step':
        setSteps((prev) => [...prev, evt])
        break
      case 'token':
        setAnswer((prev) => prev + evt.text)
        break
      case 'sources':
        setSources(evt.sources || [])
        break
      case 'error':
        alert('Error del agente: ' + evt.message)
        break
      default:
        break
    }
  }

  const askAi = async (e) => {
    e.preventDefault()
    if (!question) return
    setLoading(true)
    setSteps([])
    setAnswer('')
    setSources([])

    try {
      const res = await fetch('http://localhost:8000/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data:')) continue
          handleEvent(JSON.parse(line.slice(5).trim()))
        }
      }
    } catch (err) {
      alert('Error: ¿está el backend corriendo en http://localhost:8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h1 className="text-4xl font-extrabold text-blue-900 mb-2">Bitovi AI Expert</h1>
          <p className="text-slate-500 font-medium">Expert insights from Bitovi's Blog</p>
        </div>

        <form onSubmit={askAi} className="flex gap-3 mb-6">
          <input
            type="text"
            className="flex-1 p-4 rounded-xl border-2 border-slate-100 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-lg"
            placeholder="Ej: What are the benefits of using Playwright?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-8 rounded-xl font-bold text-lg hover:bg-blue-700 active:scale-[0.98] transition-all disabled:bg-slate-300"
          >
            {loading ? 'Pensando...' : 'Consultar'}
          </button>
        </form>

        <div className="grid grid-cols-1 md:grid-cols-[320px_1fr] gap-6">
          {/* SIDE PANEL: agent flow */}
          <aside className="bg-white rounded-2xl shadow-lg p-5 border border-slate-100 h-fit">
            <h2 className="text-xs font-black text-slate-400 uppercase tracking-wider mb-4">
              Flujo del agente
            </h2>
            {steps.length === 0 && !loading && (
              <p className="text-sm text-slate-400">Los pasos aparecerán acá.</p>
            )}
            <ol className="space-y-3">
              {steps.map((s, i) => (
                <li key={i} className="text-sm">
                  <div className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <div className="flex-1">
                      <span className="text-slate-700 font-medium">{s.label}</span>
                      <StepDetail detail={s.detail} />
                    </div>
                  </div>
                </li>
              ))}
              {loading && (
                <li className="text-sm flex items-center gap-2 text-blue-500 animate-pulse">
                  <span>⟳</span> <span>en curso…</span>
                </li>
              )}
            </ol>
          </aside>

          {/* MAIN: streaming answer */}
          <main className="bg-white rounded-2xl shadow-xl p-8 border border-slate-100 min-h-[200px]">
            {!answer && !loading && (
              <p className="text-slate-400">La respuesta aparecerá acá, escribiéndose en vivo.</p>
            )}
            {answer && (
              <div className="prose prose-blue max-w-none text-slate-700">
                <ReactMarkdown components={markdownComponents}>{answer}</ReactMarkdown>
              </div>
            )}

            {sources.length > 0 && (
              <div className="mt-8">
                <h3 className="text-xs font-bold text-slate-400 uppercase mb-3">Fuentes citadas:</h3>
                <div className="flex flex-col gap-3">
                  {sources.map((s, i) => (
                    <a key={i} href={s.url} target="_blank" rel="noreferrer"
                      className="group block bg-white p-4 border border-slate-200 rounded-xl shadow-sm hover:border-blue-400 hover:shadow-md transition-all">
                      <span className="text-blue-600 font-bold text-sm group-hover:underline">
                        {s.title || 'Documento de Bitovi'}
                      </span>
                      <div className="flex justify-between items-center mt-2">
                        <span className="text-slate-500 text-[11px] font-medium">👤 {s.author || 'Equipo Bitovi'}</span>
                        <span className="text-slate-300 text-[10px] truncate max-w-[150px]">{s.url}</span>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </main>
        </div>

        <footer className="mt-8 text-center text-slate-400 text-sm">
          Built with LangGraph + Ollama + React
        </footer>
      </div>
    </div>
  )
}

export default App
```

- [ ] **Step 2: Verify the frontend builds**

Run:
```bash
cd app/bitovi-frontend && npm run build && cd ../..
```
Expected: build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add app/bitovi-frontend/src/App.jsx
git commit -m "feat: two-column UI with live agent-flow panel and token streaming"
```

---

## Task 5: End-to-end verification

**Files:** none (manual verification).

- [ ] **Step 1: Ensure Ollama, backend, and frontend are running**

- Ollama reachable: `curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags` → `200`.
- Backend running on `:8000` (restarted after Task 3).
- Frontend running on `:5173` (Vite picks up the change via HMR; restart if needed).

- [ ] **Step 2: Test a synthesis query in the browser**

Open `http://localhost:5173`, ask **"What is RAG?"**.
Expected: side panel fills with steps (🔍 → 🧭 → ✨ → 📚 → ✍️), the answer writes token by token in the main area, and "Fuentes citadas" appears below.

- [ ] **Step 3: Test a listing query**

Ask **"cuál es el último blog de bitovi?"**.
Expected: steps show 🧭 with route `Fast`; 🗂️ "Armando el listado" appears; the article card renders (delivered as a single block — listing has no LLM tokens).

- [ ] **Step 4: Verify each step's "ver detalle" toggle**

Click `▸ ver detalle` on the 🧭 and 📚 steps. Expected: shows route/top_k and strategy/doc-count respectively.

- [ ] **Step 5: Final commit (if any tweaks were needed)**

```bash
git add -A
git commit -m "chore: verify live agent-flow streaming end-to-end"
```

---

## Self-Review

- **Spec coverage:** side panel (Task 4) ✓; friendly phases + collapsible detail (`build_step` Task 2 + `StepDetail` Task 4) ✓; token streaming (Task 3 `messages` mode + Task 4 `token` handler) ✓; steps stay visible in left column (Task 4) ✓; SSE protocol `step`/`token`/`sources`/`done`/`error` (Task 3) ✓; `/ask` kept + shared helpers (Task 3) ✓; no-token listing/fallback path (Task 3 Step 5) ✓; UTF-8 stdout untouched ✓; Langfuse via `_build_config` ✓; unit tests for pure functions (Task 2) ✓.
- **Spec deviation (intentional):** `generator`/`listing_generator` steps are emitted by the streaming loop, not `build_step`, for correct ordering. Documented in File Structure note and tested in `test_build_step_returns_none_for_internal_and_generator_nodes`.
- **Spec deviation (data availability):** the `tools` step detail shows strategy/doc-count/sort (from `metadata_log`) rather than BM25 scores, which are only printed, not returned in the tool payload.
- **Type consistency:** `build_step(node_name, update)`, `_format_sse(payload)`, `_sources_from_tool_content(content)`, `_build_config()`, `_GEN_LABELS` used consistently across tasks. Event shapes (`type`/`id`/`label`/`detail`/`text`/`sources`/`message`) match between backend emit (Task 3) and frontend `handleEvent` (Task 4).
- **Placeholder scan:** none — all steps contain concrete code/commands.
