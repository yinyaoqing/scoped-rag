"""Split text into overlapping chunks that remember where they came from.

Offsets are kept so an answer can point back at a position in the original file,
and so a re-ingest of an edited document can be compared against the old one.
Splitting prefers a sentence boundary near the target size: Chinese text has no
spaces, so a purely length-based cut lands mid-word far more often than in English.
"""

from __future__ import annotations

import re

from ..store.repo import ChunkSpec

# Sentence enders in both scripts, kept with the sentence they close.
_BOUNDARY = re.compile(r"[。！？!?\n]|(?<=[.;])\s")

DEFAULT_SIZE = 400
DEFAULT_OVERLAP = 60


def _boundaries(text: str) -> list[int]:
    return [m.end() for m in _BOUNDARY.finditer(text)]


def chunk_text(
    text: str, *, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[ChunkSpec]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    text = text.strip()
    if not text:
        return []

    ends = _boundaries(text)
    specs: list[ChunkSpec] = []
    start = 0
    ordinal = 0
    while start < len(text):
        target = start + size
        if target >= len(text):
            end = len(text)
        else:
            # Nearest sentence end at or before the target, as long as it makes
            # progress; otherwise fall back to a hard cut at the target.
            candidates = [e for e in ends if start < e <= target]
            end = candidates[-1] if candidates else target
        body = text[start:end].strip()
        if body:
            specs.append(ChunkSpec(ord=ordinal, text=body, start_offset=start, end_offset=end))
            ordinal += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return specs
