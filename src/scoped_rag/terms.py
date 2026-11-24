"""Terms for the keyword leg: what a chunk is indexed under and what a query asks for.

One function, used at ingest and at query time, so the two can never disagree.

- Latin runs are lowercased. A compound identifier (``FIN-204``, ``v2.3``, ``E_4031``)
  is kept whole *and* split into its parts, because people search for both the full
  code and the bare number.
- Chinese has no spaces, so a CJK run becomes overlapping character bigrams, the
  standard baseline when no word segmenter is available. A lone character stays a
  unigram rather than vanishing.

Exact identifiers are the reason this leg exists: an embedding model places
``FIN-204`` and ``FIN-240`` almost on top of each other, a term index does not.
"""

from __future__ import annotations

import re
from collections import Counter

_RUN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[一-鿿]+")
_PART = re.compile(r"[a-z0-9]+")


def terms(text: str) -> list[str]:
    out: list[str] = []
    for run in _RUN.findall(text.lower()):
        if "一" <= run[0] <= "鿿":
            if len(run) == 1:
                out.append(run)
            else:
                out.extend(run[i : i + 2] for i in range(len(run) - 1))
            continue
        out.append(run)
        parts = _PART.findall(run)
        if len(parts) > 1:
            out.extend(parts)
    return out


def term_counts(text: str) -> Counter[str]:
    return Counter(terms(text))
