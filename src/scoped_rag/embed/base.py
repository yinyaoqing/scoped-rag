"""The embedding interface.

Two implementations ship: a deterministic fake for tests and evals, and a local
sentence-transformers model for real use. A cloud provider would add a third file
and change nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier stored next to every vector, so models never get mixed."""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
