"""Deterministic embeddings with no model and no network.

Tests and the leakage eval need vectors that are stable across runs and machines.
Hashing token trigrams into a fixed number of buckets gives that, and keeps enough
lexical signal that a query about "parental leave" lands nearer the document about
parental leave than the one about server capacity -- which is all the eval needs.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

_WORD = re.compile(r"[\w一-鿿]+")


def _tokens(text: str) -> list[tuple[str, float]]:
    """Tokens with weights. A whole word is strong evidence; a CJK character bigram
    is weak evidence, because Chinese runs have no spaces and bigrams overlap between
    unrelated phrases (時間 appears in both 上班時間 and 退相干時間)."""
    out: list[tuple[str, float]] = []
    for w in _WORD.findall(text.lower()):
        out.append((w, 1.0))
        if len(w) > 1 and re.search(r"[一-鿿]", w):
            out.extend((w[i : i + 2], 0.35) for i in range(len(w) - 1))
    return out


class FakeEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fake-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok, weight in _tokens(text):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "big") % self._dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[bucket] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
