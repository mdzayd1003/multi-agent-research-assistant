import json

from research_assistant.trace import Tracer, read_trace


def test_events_are_written_as_they_happen(tmp_path):
    tracer = Tracer(tmp_path / "t.jsonl")
    tracer.emit("run_start", "orchestrator", question="q")
    assert len(read_trace(tmp_path / "t.jsonl")) == 1
    tracer.emit("run_end", "orchestrator")
    assert len(read_trace(tmp_path / "t.jsonl")) == 2


def test_every_line_is_valid_json_with_a_run_id_and_sequence(tmp_path):
    tracer = Tracer(tmp_path / "t.jsonl")
    tracer.llm_call("researcher", "prompt", "response")
    tracer.tool_call("researcher", "search_corpus", {"query": "x"}, [{"citation": "a#0"}])
    tracer.message("researcher", "orchestrator", "found it")
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert {r["run_id"] for r in rows} == {tracer.run_id}
    assert rows[1]["tool"] == "search_corpus" and rows[1]["ok"] is True


def test_long_results_are_truncated_with_a_marker(tmp_path):
    tracer = Tracer(tmp_path / "t.jsonl")
    event = tracer.tool_call("a", "t", {}, "x" * 5000)
    assert event.payload["result"].endswith("more chars]")


def test_a_tracer_without_a_path_still_records_in_memory():
    tracer = Tracer()
    tracer.emit("message", "a", content="hi")
    assert tracer.counts() == {"message": 1}


def test_parse_failures_keep_the_raw_reply(tmp_path):
    tracer = Tracer(tmp_path / "t.jsonl")
    tracer.parse_failure("analyst", "no JSON object found", "I refuse")
    row = read_trace(tmp_path / "t.jsonl")[0]
    assert row["kind"] == "parse_failure" and row["raw"] == "I refuse"
