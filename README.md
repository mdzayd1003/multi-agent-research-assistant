# 01 — Multi-Agent Research Assistant with RAG

**[▶ Live demo](https://mdzayd1003.github.io/multi-agent-research-assistant/)** — interactive results viewer, runs entirely in the browser.

An orchestrator that splits a research question into subtasks, hands them to
specialist agents over a single JSON tool-calling contract, and assembles a
cited answer. The retrieval side is written from scratch rather than imported,
so every stage — splitting, embedding, indexing, top-k — is inspectable.

Runs fully locally against [Ollama](https://ollama.com), or with no model at all
via a deterministic fake backend.

```
python -m pytest -q          # 82 tests, offline, ~1 s
make quick                   # full pipeline on the sample corpus, no model needed
```

## What is actually in here

**The retrieval pipeline** (`splitter.py`, `embeddings.py`, `vectorstore.py`,
`ingest.py`)

- `RecursiveTextSplitter` walks separators coarse-to-fine (blank line → newline
  → sentence punctuation → word → hard cut), packs pieces greedily to
  `chunk_size`, then re-overlaps by `chunk_overlap` so a sentence spanning a
  boundary stays retrievable from both sides.
- `HashingEmbedder` is the default: word unigrams and character 4-grams hashed
  into signed buckets, `log1p`-scaled, L2 normalised. No download, no network —
  which is exactly what makes the offline mode and the test suite possible.
  `SentenceTransformerEmbedder` swaps in MiniLM when the optional extra is
  installed.
- `VectorStore` is an exact cosine search over a float32 matrix (unit-norm
  vectors, so cosine is a matmul), using `argpartition` to avoid sorting the
  whole corpus for a top-4. `FaissVectorStore` (`IndexFlatIP`) is picked
  automatically at ≥2000 chunks, which is roughly where exact search stops being
  free.

**The agent layer** (`schema.py`, `agents/`, `orchestrator.py`)

- One JSON contract for all four agents: `{"thought", "tool_calls", "final"}`.
  The agent loop is written once; roles differ only in system prompt and in
  which tools they are shown.
- **researcher** searches before answering and carries citation tags;
  **analyst** reasons over what was found and does arithmetic through the
  calculator tool; **writer** assembles the answer without adding facts;
  **critic** returns `{"approved", "issues"}`.
- The orchestrator runs plan → dispatch → write → critique → *at most one*
  revise. An uncapped critic loop drifts and costs more; one pass catches the
  common failure, which is a draft that cites nothing.

**Tools** (`tools.py`) — `search_corpus`, `calculator` (an AST walker over
`+ - * / // % **`, not `eval`), `utc_now`, `web_search` (DuckDuckGo instant
answers, refused under `--offline`). Unknown tools, missing arguments and
invented arguments all come back as `ToolError` observations the agent can see,
never as a traceback.

**Tracing** (`trace.py`) — every LLM call, tool call and agent-to-agent message
is appended to JSONL with a run id and sequence number, flushed and fsynced per
line, so a crashed run still leaves a readable trace.

## Two design decisions worth arguing about

**Parse failures are never retried.** A reply that breaks the contract is logged
as a `parse_failure` and ends that agent's turn. Retrying until the model
produces valid JSON hides the thing most worth knowing: how often it can hold a
format. The count surfaces in every run report.

**The hashing embedder is the default, not a fallback.** It is lexical and
blunt, and on a small corpus it is good enough to make the orchestration
observable. Making the *offline* path the default one means the tests exercise
the same code path a user gets, instead of a mocked stand-in.

## Running it

```bash
pip install -r requirements.txt

# 1. index a corpus (any folder of .md/.txt/.py/.json)
python -m research_assistant ingest sample_corpus

# 2. look at retrieval on its own, no LLM in the loop
python -m research_assistant retrieve "how are parse failures handled?"

# 3. ask, with no model at all
python -m research_assistant --offline ask "How does top-k retrieval work?" --fake-llm

# 4. ask, against a real local model
ollama pull llama3.1:8b
python -m research_assistant ask "How does top-k retrieval work?" --json
```

`--json` prints the whole run report: the plan, each subtask result, the
critique, and the parse-failure and tool-error counts.

### Streamlit UI

```bash
python -m research_assistant serve     # http://localhost:8501
```

### Docker

```bash
docker compose up --build              # Ollama + model pull + UI on :8501
```

`docker-compose.yml` starts Ollama, waits for it to be healthy, pulls
`llama3.1:8b` in a one-shot container, and only then starts the UI.

### Optional extras

```bash
pip install -r requirements-optional.txt      # MiniLM + FAISS
python -m research_assistant --embedder minilm ingest sample_corpus
```

## Configuration

Every flag has an environment variable (`config.py`): `RA_INDEX_DIR`,
`RA_EMBEDDER`, `RA_LLM_BACKEND`, `RA_MODEL`, `RA_OLLAMA_HOST`, `RA_STORE`,
`RA_CHUNK_SIZE`, `RA_CHUNK_OVERLAP`, `RA_TOP_K`, `RA_MAX_STEPS`, `RA_OFFLINE`.

## Tests

```
python -m pytest -q
```

82 tests, no network, no model, about a second. They cover chunk-boundary and
overlap behaviour, embedding determinism and relative ranking, index round-trip
through disk, every contract violation the parser is supposed to reject, tool
argument validation, the researcher's search-before-answer ordering, the
step-budget and parse-failure paths, the single revise pass, trace ordering and
durability, and the CLI end to end.

## Layout

```
research_assistant/
├── splitter.py        recursive character splitting
├── embeddings.py      hashing embedder + optional MiniLM
├── vectorstore.py     numpy cosine store + optional FAISS
├── ingest.py          corpus → chunks → vectors → index; Retriever
├── schema.py          the JSON contract and its parser
├── llm.py             Ollama client + deterministic fake backend
├── tools.py           tool registry with argument validation
├── trace.py           JSONL tracing
├── orchestrator.py    plan → dispatch → write → critique → revise
├── agents/            the shared loop + four role prompts
├── app.py             Streamlit UI
└── __main__.py        ingest / retrieve / ask / serve
```

## Limits

- The hashing embedder is lexical. A question phrased with none of the corpus's
  words will retrieve poorly; MiniLM is the fix and it is one flag away.
- `web_search` uses DuckDuckGo's instant-answer endpoint, which covers
  encyclopaedic queries and little else. It is a demonstration of the tool
  contract, not a research-grade search tool.
- Subtasks run sequentially. They are independent and could be parallelised;
  with a single local model serving them there is nothing to gain.

---

One of seven projects. Index: [mdzayd1003/zaid-portfolio-projects](https://github.com/mdzayd1003/zaid-portfolio-projects)
