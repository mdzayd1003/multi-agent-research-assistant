"""A small multi-agent research assistant with retrieval augmented generation.

Everything here is written to run offline: the default embedder is a hashing
embedder that needs no model download, the default vector store is numpy, and
``--fake-llm`` swaps in a deterministic backend so the orchestration logic can
be exercised without a model server.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
