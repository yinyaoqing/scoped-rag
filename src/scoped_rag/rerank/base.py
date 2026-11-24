"""The re-ranker interface.

A re-ranker reads the query and one chunk together and scores how well the chunk
answers it. That is slower and sharper than comparing two vectors made separately,
so it runs last, on the few rows retrieval already found. It is handed text the
asker is permitted to read and nothing else: it has no access to the store.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score per text, higher is more relevant."""
