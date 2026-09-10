import json

from research_assistant.__main__ import main


def test_ingest_then_retrieve_then_ask(tmp_path, corpus_dir, capsys):
    index = tmp_path / "index"

    assert main(["--index-dir", str(index), "ingest", str(corpus_dir), "--store", "numpy"]) == 0
    assert "indexed" in capsys.readouterr().out
    assert (index / "chunks.json").exists()

    assert main(["--index-dir", str(index), "retrieve", "how are parse failures handled?"]) == 0
    assert "agents.md#" in capsys.readouterr().out

    trace = tmp_path / "trace.jsonl"
    assert main(
        ["--index-dir", str(index), "--offline", "ask", "how does top-k search work?", "--fake-llm", "--trace", str(trace)]
    ) == 0
    out = capsys.readouterr().out
    assert "sources:" in out
    assert trace.exists() and trace.read_text().strip()


def test_ask_can_emit_the_whole_report_as_json(tmp_path, corpus_dir, capsys):
    index = tmp_path / "index"
    main(["--index-dir", str(index), "ingest", str(corpus_dir)])
    capsys.readouterr()

    main(["--index-dir", str(index), "ask", "what does the critic do?", "--fake-llm", "--json", "--trace", str(tmp_path / "t.jsonl")])
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] and payload["plan"] and payload["subtasks"]


def test_asking_without_an_index_fails_with_a_useful_message(tmp_path):
    try:
        main(["--index-dir", str(tmp_path / "missing"), "ask", "q", "--fake-llm"])
    except SystemExit as exc:
        assert "run `ingest` first" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")
