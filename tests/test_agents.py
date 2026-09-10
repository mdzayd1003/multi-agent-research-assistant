import json

import pytest

from research_assistant.agents import AnalystAgent, CriticAgent, ResearcherAgent, WriterAgent
from research_assistant.llm import FakeLLM
from research_assistant.tools import default_registry


@pytest.fixture()
def registry(retriever):
    return default_registry(retriever, offline=True)


def test_the_researcher_searches_before_answering(registry, tracer):
    result = ResearcherAgent(FakeLLM(), registry, tracer).run("how are parse failures handled?")
    assert result.ok
    assert result.citations
    kinds = [e.kind for e in tracer.events]
    assert kinds.index("tool_call") < kinds.index("message")


def test_the_researcher_carries_retrieved_citations_into_its_answer(registry, tracer):
    result = ResearcherAgent(FakeLLM(), registry, tracer).run("how is the trace written?")
    tool_events = [e for e in tracer.events if e.kind == "tool_call"]
    retrieved = {r["citation"] for r in tool_events[0].payload["result"]}
    assert set(result.citations) <= retrieved


def test_a_role_only_sees_the_tools_it_is_allowed(registry, tracer):
    system = WriterAgent(FakeLLM(), registry, tracer).system_prompt()
    assert "search_corpus" not in system
    assert "ROLE: writer" in system


def test_an_unparseable_reply_is_recorded_and_never_retried(registry, tracer):
    llm = FakeLLM(scripted=lambda system, prompt: "I refuse to emit JSON")
    result = AnalystAgent(llm, registry, tracer).run("anything")
    assert not result.ok
    assert result.parse_failures == 1
    assert len(llm.calls) == 1, "a malformed reply must not be retried"
    assert [e.kind for e in tracer.events if e.kind == "parse_failure"]


def test_a_tool_error_becomes_an_observation_rather_than_a_crash(registry, tracer):
    replies = iter(
        [
            json.dumps({"thought": "t", "tool_calls": [{"name": "nope", "arguments": {}}]}),
            json.dumps({"thought": "t", "tool_calls": [], "final": {"answer": "recovered", "citations": []}}),
        ]
    )
    llm = FakeLLM(scripted=lambda system, prompt: next(replies))
    result = AnalystAgent(llm, registry, tracer).run("q")
    assert result.tool_errors == 1
    assert result.answer == "recovered"
    assert result.steps == 2


def test_an_agent_that_never_finishes_hits_the_step_budget(registry, tracer):
    llm = FakeLLM(scripted=lambda system, prompt: json.dumps({"thought": "t", "tool_calls": [{"name": "utc_now"}]}))
    result = AnalystAgent(llm, registry, tracer, max_steps=2).run("q")
    assert not result.ok
    assert result.steps == 2
    assert "budget" in result.answer


def test_observations_are_fed_back_into_the_next_prompt(registry, tracer):
    llm = FakeLLM()
    ResearcherAgent(llm, registry, tracer).run("how does top-k search work?")
    assert "OBSERVATIONS:" in llm.calls[-1]


def test_the_critic_rejects_a_draft_with_no_citation(registry, tracer):
    result = CriticAgent(FakeLLM(), registry, tracer, max_steps=1).run("q", context='DRAFT:\n{"draft": "trust me"}')
    assert result.raw_final["approved"] is False
    assert result.raw_final["issues"]


def test_the_critic_approves_a_cited_draft(registry, tracer):
    result = CriticAgent(FakeLLM(), registry, tracer, max_steps=1).run(
        "q", context='DRAFT:\n{"draft": "supported [agents.md#1]"}'
    )
    assert result.raw_final["approved"] is True
