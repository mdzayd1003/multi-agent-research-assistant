import pytest

from research_assistant.tools import ToolError, default_registry


@pytest.fixture()
def registry(retriever):
    return default_registry(retriever, offline=True)


def test_corpus_search_returns_scored_citations(registry):
    results = registry.call("search_corpus", {"query": "how is the trace written", "k": 2})
    assert len(results) == 2
    assert all("#" in r["citation"] for r in results)
    assert results[0]["score"] >= results[1]["score"]


def test_calculator_evaluates_arithmetic(registry):
    assert registry.call("calculator", {"expression": "(2 + 3) * 4 - 1"}) == 19.0


def test_calculator_refuses_anything_but_arithmetic(registry):
    with pytest.raises(ToolError):
        registry.call("calculator", {"expression": "__import__('os').listdir('.')"})


def test_calculator_reports_division_by_zero(registry):
    with pytest.raises(ToolError):
        registry.call("calculator", {"expression": "1/0"})


def test_web_search_is_refused_in_offline_mode(registry):
    with pytest.raises(ToolError, match="offline"):
        registry.call("web_search", {"query": "anything"})


def test_an_unknown_tool_is_an_error_not_a_traceback(registry):
    with pytest.raises(ToolError, match="no such tool"):
        registry.call("summon_daemon", {})


def test_a_missing_required_argument_is_reported(registry):
    with pytest.raises(ToolError, match="missing required"):
        registry.call("search_corpus", {"k": 2})


def test_an_invented_argument_is_reported(registry):
    with pytest.raises(ToolError, match="unexpected"):
        registry.call("utc_now", {"timezone": "CET"})


def test_the_tool_spec_lists_every_tool(registry):
    described = registry.describe()
    for name in ("search_corpus", "calculator", "utc_now", "web_search"):
        assert f"- {name}(" in described
