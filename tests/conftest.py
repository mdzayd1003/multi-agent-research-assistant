import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_assistant.embeddings import HashingEmbedder  # noqa: E402
from research_assistant.ingest import Retriever, build_index  # noqa: E402
from research_assistant.trace import Tracer  # noqa: E402


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return ROOT / "sample_corpus"


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture()
def retriever(corpus_dir, embedder) -> Retriever:
    store = build_index(corpus_dir, embedder=embedder, chunk_size=500, chunk_overlap=80, backend="numpy")
    return Retriever(store, embedder, default_k=4)


@pytest.fixture()
def tracer(tmp_path) -> Tracer:
    return Tracer(tmp_path / "trace.jsonl")
