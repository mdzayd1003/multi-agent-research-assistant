"""Recursive character text splitting.

Written from scratch rather than pulled from LangChain so the chunk boundaries
are inspectable: the splitter walks a list of separators from coarse to fine
and only falls back to a hard character cut when no separator helps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class RecursiveTextSplitter:
    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: List[str] = field(default_factory=lambda: list(DEFAULT_SEPARATORS))

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.separators:
            raise ValueError("at least one separator is required")

    # -- public API ---------------------------------------------------------
    def split(self, text: str) -> List[str]:
        """Split ``text`` into chunks of at most ``chunk_size`` characters."""
        pieces = self._split_recursive(text.strip(), self.separators)
        return self._merge(pieces)

    # -- internals ----------------------------------------------------------
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            # No separator left: hard cut on the character grid.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = _split_keeping_separator(text, separator)
        if len(parts) == 1:
            return self._split_recursive(text, remaining)

        out: List[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                if part:
                    out.append(part)
            else:
                out.extend(self._split_recursive(part, remaining))
        return out

    def _merge(self, pieces: List[str]) -> List[str]:
        """Greedily pack pieces up to chunk_size, then re-overlap the result."""
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            candidate = current + piece if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
            current = piece if len(piece) <= self.chunk_size else piece[: self.chunk_size]
        if current.strip():
            chunks.append(current.strip())

        if self.chunk_overlap == 0 or len(chunks) < 2:
            return [c for c in chunks if c]

        overlapped = [chunks[0]]
        for previous, chunk in zip(chunks, chunks[1:]):
            tail = previous[-self.chunk_overlap :]
            merged = (tail + " " + chunk).strip()
            overlapped.append(merged[: self.chunk_size + self.chunk_overlap])
        return [c for c in overlapped if c]


def _split_keeping_separator(text: str, separator: str) -> List[str]:
    """Split on ``separator`` but keep it attached to the preceding part."""
    parts = text.split(separator)
    out = [p + separator for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return [p for p in out if p]
