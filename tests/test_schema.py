import pytest

from research_assistant.schema import ParseFailure, parse_agent_reply, parse_plan


def test_a_tool_call_reply_parses():
    reply = parse_agent_reply('{"thought": "need evidence", "tool_calls": [{"name": "search_corpus", "arguments": {"query": "x"}}]}')
    assert not reply.is_final
    assert reply.tool_calls[0].name == "search_corpus"
    assert reply.tool_calls[0].arguments == {"query": "x"}


def test_a_final_reply_parses():
    reply = parse_agent_reply('{"thought": "done", "tool_calls": [], "final": {"answer": "42", "citations": ["a.md#0"]}}')
    assert reply.is_final and reply.final["answer"] == "42"


def test_json_inside_a_code_fence_is_recovered():
    reply = parse_agent_reply('Sure!\n```json\n{"thought": "t", "tool_calls": [], "final": {"answer": "a"}}\n```\nHope that helps.')
    assert reply.final["answer"] == "a"


def test_json_wrapped_in_prose_is_recovered():
    reply = parse_agent_reply('Here you go: {"thought": "t", "tool_calls": [], "final": {"answer": "a"}} done')
    assert reply.final["answer"] == "a"


def test_missing_arguments_default_to_an_empty_object():
    reply = parse_agent_reply('{"thought": "t", "tool_calls": [{"name": "utc_now"}]}')
    assert reply.tool_calls[0].arguments == {}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "no json at all",
        "{not valid json}",
        '{"thought": 12, "tool_calls": []}',
        '{"thought": "t", "tool_calls": "search"}',
        '{"thought": "t", "tool_calls": [{"arguments": {}}]}',
        '{"thought": "t", "tool_calls": [{"name": "t", "arguments": []}]}',
        '{"thought": "t", "tool_calls": [], "final": "just a string"}',
        '{"thought": "t", "tool_calls": []}',
    ],
)
def test_contract_violations_raise_parse_failure(raw):
    with pytest.raises(ParseFailure):
        parse_agent_reply(raw)


def test_parse_failure_keeps_the_raw_reply_for_the_trace():
    with pytest.raises(ParseFailure) as excinfo:
        parse_agent_reply("total nonsense")
    assert excinfo.value.raw == "total nonsense"


def test_a_plan_parses_and_defaults_the_agent():
    subtasks = parse_plan('{"subtasks": [{"question": "q1"}, {"id": "b", "question": "q2", "agent": "analyst"}]}')
    assert [s.id for s in subtasks] == ["s1", "b"]
    assert [s.agent for s in subtasks] == ["researcher", "analyst"]


@pytest.mark.parametrize(
    "raw",
    ['{"subtasks": []}', '{"subtasks": [{"question": "  "}]}', '{"plan": []}'],
)
def test_malformed_plans_raise(raw):
    with pytest.raises(ParseFailure):
        parse_plan(raw)


def test_a_plan_naming_an_unknown_agent_is_rejected():
    with pytest.raises(ParseFailure):
        parse_plan('{"subtasks": [{"question": "q", "agent": "wizard"}]}', allowed_agents=["researcher"])
