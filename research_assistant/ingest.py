"""Corpus ingestion: files -> chunks -> vectors -> a saved index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .embeddings import Embedder, build_embedder
from .splitter import RecursiveTextSplitter
from .vectorstore import Chunk, ScoredChunk, VectorStore, build_store

TEXT_SUFFIXES = {".md", ".txt", ".rst", ".py", ".json"}


def iter_documents(root: str | Path, suffixes: Optional[Iterable[str]] = None) -> List[Path]:
    root = Path(root)
    allowed = set(suffixes or TEXT_SUFFIXES)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def chunk_documents(
    paths: Sequence[Path],
    splitter: Optional[RecursiveTextSplitter] = None,
    *,
    root: Optional[Path] = None,
) -> List[Chunk]:
    splitter = splitter or RecursiveTextSplitter()
    chunks: List[Chunk] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        source = str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name
        for index, piece in enumerate(splitter.split(text)):
            chunks.append(
                Chunk(
                    id=f"{source}#{index}",
                    text=piece,
                    source=source,
                    index=index,
                    metadata={"chars": len(piece)},
                )
            )
    return chunks


def build_index(
    corpus: str | Path,
    *,
    embedder: Optional[Embedder] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    backend: str = "auto",
) -> VectorStore:
    corpus_path = Path(corpus)
    embedder = embedder or build_embedder("hashing")
    paths = iter_documents(corpus_path)
    if not paths:
        raise FileNotFoundError(f"no readable documents under {corpus_path}")
    chunks = chunk_documents(
        paths,
        RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        root=corpus_path if corpus_path.is_dir() else None,
    )
    store = build_store(embedder.dim, backend, corpus_size=len(chunks))
    vectors = embedder.encode([c.text for c in chunks])
    store.add(chunks, vectors)
    return store


@dataclass
class Retriever:
    """Embeds the query with the same embedder used at ingest time."""

    store: VectorStore
    embedder: Embedder
    default_k: int = 4

    def retrieve(self, query: str, k: Optional[int] = None) -> List[ScoredChunk]:
        if not query.strip():
            return []
        vector = self.embedder.encode([query])[0]
        return self.store.search(np.asarray(vector), k=k or self.default_k)
