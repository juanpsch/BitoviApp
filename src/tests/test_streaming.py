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
