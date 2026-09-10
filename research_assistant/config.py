"""Run configuration, resolved from CLI flags with environment fallbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    index_dir: Path = Path(_env("RA_INDEX_DIR", ".cache/index"))
    trace_dir: Path = Path(_env("RA_TRACE_DIR", "results/traces"))
    embedder: str = _env("RA_EMBEDDER", "hashing")
    llm_backend: str = _env("RA_LLM_BACKEND", "ollama")
    model: str = _env("RA_MODEL", "llama3.1:8b")
    ollama_host: str = _env("RA_OLLAMA_HOST", "http://localhost:11434")
    store_backend: str = _env("RA_STORE", "auto")
    chunk_size: int = int(_env("RA_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(_env("RA_CHUNK_OVERLAP", "120"))
    top_k: int = int(_env("RA_TOP_K", "4"))
    max_steps: int = int(_env("RA_MAX_STEPS", "3"))
    offline: bool = _env("RA_OFFLINE", "0") == "1"
