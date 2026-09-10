"""Vector store with a numpy backend and an optional FAISS backend.

The numpy backend is the default because it is exact, has no dependency beyond
numpy and is fast enough for a corpus of a few thousand chunks. FAISS is used
when it is installed and the corpus is large enough for the index to pay for
itself; both expose the same ``search`` signature so the retriever does not care.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .embeddings import l2_normalise


@dataclass
class Chunk:
    """One retrievable unit of text plus where it came from."""

    id: str
    text: str
    source: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.index}"


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """Exact cosine-similarity search over unit-norm vectors."""

    backend = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks: List[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(chunks) != vectors.shape[0]:
            raise ValueError("number of chunks and vectors must match")
        self._vectors = np.vstack([self._vectors, l2_normalise(vectors)])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[ScoredChunk]:
        if len(self) == 0:
            return []
        query = l2_normalise(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        scores = (self._vectors @ query.T).ravel()
        k = max(1, min(k, len(self)))
        # argpartition first so we do not sort the whole corpus for a top-4
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredChunk(self.chunks[int(i)], float(scores[int(i)])) for i in top]

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        payload = {
            "dim": self.dim,
            "backend": self.backend,
            "chunks": [asdict(c) for c in self.chunks],
        }
        (path / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        path = Path(directory)
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        store = cls(int(payload["dim"]))
        vectors = np.load(path / "vectors.npy")
        chunks = [Chunk(**c) for c in payload["chunks"]]
        store.add(chunks, vectors)
        return store


class FaissVectorStore(VectorStore):
    """Same contract, backed by ``faiss.IndexFlatIP`` over normalised vectors."""

    backend = "faiss"

    def __init__(self, dim: int) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("faiss-cpu is not installed") from exc
        super().__init__(dim)
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:  # pragma: no cover
        super().add(chunks, vectors)
        self._index.add(l2_normalise(np.asarray(vectors, dtype=np.float32)))

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[ScoredChunk]:  # pragma: no cover
        if len(self) == 0:
            return []
        query = l2_normalise(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        k = max(1, min(k, len(self)))
        scores, indices = self._index.search(query, k)
        return [
            ScoredChunk(self.chunks[int(i)], float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]


def build_store(dim: int, backend: str = "auto", *, corpus_size: int = 0) -> VectorStore:
    """Pick a backend. ``auto`` only reaches for FAISS on a corpus worth indexing."""
    if backend == "numpy":
        return VectorStore(dim)
    if backend == "faiss":
        return FaissVectorStore(dim)
    if backend != "auto":
        raise ValueError(f"unknown vector store backend: {backend!r}")
    if corpus_size >= 2000:
        try:
            return FaissVectorStore(dim)
        except ImportError:
            pass
    return VectorStore(dim)


def format_context(hits: Sequence[ScoredChunk], max_chars: int = 2400) -> Tuple[str, List[str]]:
    """Render retrieved chunks into a prompt block plus the citation list."""
    lines: List[str] = []
    citations: List[str] = []
    used = 0
    for hit in hits:
        block = f"[{hit.chunk.citation}] {hit.chunk.text}"
        if used + len(block) > max_chars and lines:
            break
        lines.append(block)
        citations.append(hit.chunk.citation)
        used += len(block)
    return "\n\n".join(lines), citations
