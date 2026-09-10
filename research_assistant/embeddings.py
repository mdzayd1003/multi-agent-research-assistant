"""Embedding backends.

Two of them:

``HashingEmbedder``
    A deterministic bag-of-ngrams embedder built on top of ``blake2b``. It needs
    no download and no network, which is what makes the test suite and the
    ``--offline`` path possible. It is a real (if blunt) lexical embedder:
    character n-grams and word unigrams are hashed into a fixed number of
    buckets, sublinearly weighted and L2 normalised.

``SentenceTransformerEmbedder``
    Wraps ``sentence-transformers`` (MiniLM by default). Imported lazily so the
    package stays optional.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns text into a matrix of unit-norm row vectors."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _sign(token: str) -> float:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=1, salt=b"sign").digest()
    return 1.0 if digest[0] & 1 else -1.0


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def char_ngrams(text: str, n: int) -> Iterable[str]:
    cleaned = " ".join(tokenize(text))
    if len(cleaned) < n:
        if cleaned:
            yield cleaned
        return
    for i in range(len(cleaned) - n + 1):
        yield cleaned[i : i + n]


@dataclass
class HashingEmbedder:
    """Signed-hash bag of words + character n-grams, L2 normalised."""

    dim: int = 384
    ngram: int = 4
    word_weight: float = 1.0
    ngram_weight: float = 0.5

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for word in tokenize(text):
                idx = _bucket("w:" + word, self.dim)
                counts[idx] = counts.get(idx, 0.0) + self.word_weight * _sign(word)
            if self.ngram_weight:
                for gram in char_ngrams(text, self.ngram):
                    idx = _bucket("c:" + gram, self.dim)
                    counts[idx] = counts.get(idx, 0.0) + self.ngram_weight * _sign(gram)
            for idx, value in counts.items():
                # sublinear scaling keeps a repeated word from dominating
                out[row, idx] = math.copysign(math.log1p(abs(value)), value)
        return l2_normalise(out)


class SentenceTransformerEmbedder:
    """MiniLM (or any sentence-transformers model). Requires a download."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed; "
                "install requirements-optional.txt or use the hashing embedder"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs model
        vectors = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return l2_normalise(np.asarray(vectors, dtype=np.float32))


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def build_embedder(name: str = "hashing", *, dim: int = 384) -> Embedder:
    if name in {"hashing", "offline"}:
        return HashingEmbedder(dim=dim)
    if name in {"minilm", "sentence-transformers"}:
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")
