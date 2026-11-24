"""A deterministic re-ranker with no model, for tests.

It scores a chunk by the share of the query's terms it contains. That is enough to
exercise the plumbing -- reordering, the candidate pool, the score threshold -- and
says nothing about how good a real cross-encoder is. Quality numbers in this repo
come from `LocalReranker`, never from this class.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..terms import terms


class FakeReranker:
    @property
    def name(self) -> str:
        return "fake-overlap"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        wanted = set(terms(query))
        if not wanted:
            return [0.0 for _ in texts]
        return [len(wanted & set(terms(t))) / len(wanted) for t in texts]
