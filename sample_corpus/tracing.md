# Tracing

Every LLM call, tool call and agent-to-agent message is appended to a JSONL
file as it happens, with a run id and a monotonically increasing sequence
number. Each line is flushed and fsynced, so a run that crashes still leaves a
readable trace up to the crash.

Event kinds are `run_start`, `llm_call`, `tool_call`, `message`,
`parse_failure` and `run_end`. Long strings are truncated with an explicit
marker rather than silently cut, so you can tell a short tool result from a
trimmed one.

The trace is what makes the system debuggable: without it there is no way to
tell whether the writer agent used what the researcher retrieved, or whether an
answer that looks well sourced was assembled from citations the model invented.
