import json

from research_assistant.llm import FakeLLM
from research_assistant.orchestrator import Orchestrator


def test_a_full_run_produces_a_cited_answer(retriever, tracer):
    report = Orchestrator(FakeLLM(), retriever, tracer=tracer).run("how are parse failures handled?")
    assert report.answer
    assert report.citations
    assert report.parse_failures == 0


def test_the_plan_is_recorded_and_dispatched(retriever, tracer):
    report = Orchestrator(FakeLLM(), retriever, tracer=tracer).run("how does retrieval work?")
    assert len(report.plan) == 2
    assert [s["id"] for s in report.subtask_results] == [s["id"] for s in report.plan]
    assert {s["agent"] for s in report.subtask_results} == {"researcher", "analyst"}


def test_an_unparseable_plan_degrades_to_single_agent_rag(retriever, tracer):
    def scripted(system, prompt):
        return "I cannot plan" if "ROLE: orchestrator" in system else None

    report = Orchestrator(FakeLLM(scripted=scripted), retriever, tracer=tracer).run("how does retrieval work?")
    assert len(report.plan) == 1
    assert report.plan[0]["agent"] == "researcher"
    assert report.answer
    assert any(e.kind == "parse_failure" for e in tracer.events)


def test_a_rejected_draft_triggers_exactly_one_revision(retriever, tracer):
    def scripted(system, prompt):
        if "ROLE: critic" in system:
            return json.dumps(
                {"thought": "t", "tool_calls": [], "final": {"approved": False, "issues": ["no citation"], "answer": "fix it"}}
            )
        return None

    report = Orchestrator(FakeLLM(scripted=scripted), retriever, tracer=tracer).run("how does tracing work?")
    assert report.revised is True
    assert report.critique["approved"] is False
    writer_finals = [e for e in tracer.events if e.kind == "message" and e.actor == "writer"]
    assert len(writer_finals) == 2


def test_the_trace_covers_the_whole_run(retriever, tracer):
    Orchestrator(FakeLLM(), retriever, tracer=tracer).run("how does top-k search work?")
    kinds = [e.kind for e in tracer.events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    for kind in ("llm_call", "tool_call", "message"):
        assert kind in kinds
    assert [e.seq for e in tracer.events] == list(range(1, len(tracer.events) + 1))


def test_the_report_serialises_to_json(retriever, tracer):
    report = Orchestrator(FakeLLM(), retriever, tracer=tracer).run("what does the critic do?")
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["question"] and payload["answer"]
    assert payload["run_id"] == tracer.run_id
