# Smoke output

Produced by `make quick` on the three-document `sample_corpus/`, with the
deterministic `--fake-llm` backend and `--offline`. It proves the pipeline runs
end to end — plan, dispatch, retrieve, write, critique — and that citations
survive from the retriever to the final answer.

It is **not** a quality result. The fake backend does not reason; it echoes the
role and the citations it was handed. Numbers from a real model go in the README
only after a full run.

- `report.json` — the full run report (plan, per-subtask results, critique)
- `trace.jsonl` — every LLM call, tool call and agent message from that run
