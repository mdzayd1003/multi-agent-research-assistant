# The agent layer

## One contract

Every agent speaks the same JSON contract: a `thought`, a list of `tool_calls`,
and a `final` object that is null until the agent is done. Having one contract
for all four agents means the agent loop is written once and the roles differ
only in their system prompt.

## The four roles

The **researcher** gathers evidence. It is expected to call `search_corpus`
before answering and to carry citation tags into its reply. The **analyst**
reasons over evidence the researcher collected and uses the calculator tool for
arithmetic. The **writer** turns findings into one coherent answer without
introducing new facts. The **critic** checks the draft against the evidence and
returns `{"approved": bool, "issues": [...]}`.

## Parse failures are data

A reply that does not satisfy the contract raises `ParseFailure`, which is
logged to the trace and ends that agent's turn. It is never retried. Retrying a
malformed reply hides the exact behaviour worth measuring: how often the model
can hold a format under pressure.

## The revise loop is capped

The orchestrator runs plan, dispatch, write, critique, and at most one revise
pass. An uncapped critic loop drifts and costs more; one pass catches the common
failure, which is a draft that cites nothing.
