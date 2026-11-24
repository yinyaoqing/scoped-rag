"""Permission-scoped retrieval: a vector leg, a keyword leg, their fusion, a re-ranker.

The point of this file is that the access predicate goes to the database together
with the ranking: WHERE filters, ORDER BY ranks what survived, LIMIT takes k.
Filtering afterwards would both leak (result counts reveal documents you cannot
read) and rank worse (your top k gets eaten by rows you never get to see).

That rule holds for every stage added on top of the vector search:

- The keyword leg carries the same predicate, and goes one step further: BM25's
  corpus statistics (document frequency, average length) are computed over the
  asker's readable set inside the same query. Corpus-wide statistics would let a
  hidden document change the scores of visible ones, which turns ranking into a
  way of asking what the hidden documents contain.
- Fusion and re-ranking only ever see rows the two legs returned, so they can
  reorder what the asker may read and can never add to it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..embed.base import Embedder
from ..rerank.base import Reranker
from ..store.repo import Store
from ..terms import terms
from .acl import readable_predicate

MODES = ("vector", "keyword", "hybrid")

# BM25 defaults from the literature; not tuned against the eval set.
BM25_K1 = 1.2
BM25_B = 0.75
# Reciprocal rank fusion constant from Cormack et al. (2009); also not tuned.
RRF_K = 60


class TermIndexMissing(RuntimeError):
    """The database predates the keyword leg. Failing beats quietly returning vector-only
    results under a mode that claims to be hybrid."""


@dataclass(frozen=True)
class Hit:
    """`distance` is always the cosine distance to the query, whichever leg found the
    row, so the answer path can apply one relevance threshold. The other scores are
    filled in by the stage that produced them and are `None` otherwise."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    distance: float
    keyword: float | None = None
    fused: float | None = None
    rerank: float | None = None


def _scope(store: Store, asker: str) -> tuple[str, dict[str, str]]:
    """The one place retrieval obtains the access predicate."""
    return readable_predicate(store.d, asker, is_admin=store.is_admin(asker), doc_alias="d")


def search(
    store: Store,
    embedder: Embedder,
    *,
    asker: str,
    query: str,
    k: int = 5,
) -> list[Hit]:
    """The vector leg."""
    if k <= 0:
        raise ValueError("k must be positive")
    acl_sql, acl_params = _scope(store, asker)
    p = store.d.ph
    distance = store.d.distance("e.vector", "q")
    sql = (
        f"SELECT c.id, c.document_id, d.title, c.text, {distance} AS distance "
        "FROM chunks c "
        "JOIN documents d ON d.id = c.document_id "
        "JOIN embeddings e ON e.chunk_id = c.id "
        f"WHERE e.embedder = {p('embedder')} AND {acl_sql} "
        f"ORDER BY distance ASC LIMIT {p('k')}"
    )
    params: dict[str, object] = {
        "q": store.d.encode(list(embedder.embed_query(query))),
        "embedder": embedder.name,
        "k": k,
        **acl_params,
    }
    return [Hit(r[0], r[1], r[2], r[3], float(r[4])) for r in store.execute(sql, params)]


def keyword_search(
    store: Store,
    embedder: Embedder,
    *,
    asker: str,
    query: str,
    k: int = 5,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[Hit]:
    """The keyword leg: BM25 over the asker's readable chunks, in one query.

    `readable` is the scoped population. Every statistic below is derived from it:
    `n` and `avgdl` in `stats`, per-term document frequency in `df`. A document the
    asker cannot read therefore contributes to nothing -- not to the results and not
    to the numbers the results are ranked by.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    wanted = list(dict.fromkeys(terms(query)))
    if not wanted:
        return []
    acl_sql, acl_params = _scope(store, asker)
    p = store.d.ph
    distance = store.d.distance("e.vector", "q")
    term_params = {f"t{i}": term for i, term in enumerate(wanted)}
    in_list = ", ".join(p(name) for name in term_params)
    sql = (
        "WITH readable AS ("
        "  SELECT c.id, c.document_id, d.title, c.text"
        "  FROM chunks c"
        "  JOIN documents d ON d.id = c.document_id"
        "  JOIN embeddings e ON e.chunk_id = c.id"
        f"  WHERE e.embedder = {p('embedder')} AND {acl_sql}"
        "), lens AS ("
        "  SELECT t.chunk_id, SUM(t.tf) AS n_terms"
        "  FROM chunk_terms t JOIN readable r ON r.id = t.chunk_id"
        "  GROUP BY t.chunk_id"
        "), stats AS ("
        "  SELECT COUNT(*) AS n, AVG(n_terms) AS avgdl FROM lens"
        "), df AS ("
        "  SELECT t.term, COUNT(*) AS df"
        "  FROM chunk_terms t JOIN readable r ON r.id = t.chunk_id"
        f"  WHERE t.term IN ({in_list})"
        "  GROUP BY t.term"
        ") "
        f"SELECT r.id, r.document_id, r.title, r.text, {distance} AS distance,"
        "  SUM(LN(1 + (s.n - f.df + 0.5) / (f.df + 0.5))"
        f"      * (t.tf * ({p('k1')} + 1))"
        f"      / (t.tf + {p('k1')} * (1 - {p('b')} + {p('b')} * l.n_terms / s.avgdl))) AS score "
        "FROM chunk_terms t "
        "JOIN readable r ON r.id = t.chunk_id "
        "JOIN lens l ON l.chunk_id = r.id "
        "JOIN df f ON f.term = t.term "
        "JOIN embeddings e ON e.chunk_id = r.id "
        "CROSS JOIN stats s "
        f"WHERE t.term IN ({in_list}) "
        "GROUP BY r.id, r.document_id, r.title, r.text, e.vector "
        f"ORDER BY score DESC, r.id ASC LIMIT {p('k')}"
    )
    params: dict[str, object] = {
        "q": store.d.encode(list(embedder.embed_query(query))),
        "embedder": embedder.name,
        "k": k,
        "k1": float(k1),
        "b": float(b),
        **term_params,
        **acl_params,
    }
    return [
        Hit(r[0], r[1], r[2], r[3], float(r[4]), keyword=float(r[5]))
        for r in store.execute(sql, params)
    ]


def fuse(vector: list[Hit], keyword: list[Hit], *, rrf_k: int = RRF_K) -> list[Hit]:
    """Reciprocal rank fusion. Ranks, not scores, because cosine distance and BM25
    are on unrelated scales and any weighting between them would be a tuned guess."""
    by_id: dict[str, Hit] = {}
    score: dict[str, float] = {}
    for hits in (vector, keyword):
        for rank, hit in enumerate(hits, start=1):
            score[hit.chunk_id] = score.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            known = by_id.get(hit.chunk_id)
            by_id[hit.chunk_id] = hit if known is None else replace(known, keyword=hit.keyword)
    ordered = sorted(score, key=lambda cid: (-score[cid], cid))
    return [replace(by_id[cid], fused=score[cid]) for cid in ordered]


def retrieve(
    store: Store,
    embedder: Embedder,
    *,
    asker: str,
    query: str,
    k: int = 5,
    mode: str = "vector",
    reranker: Reranker | None = None,
    candidates: int | None = None,
) -> list[Hit]:
    """Retrieve k chunks the asker may read.

    `candidates` is how deep each leg goes before fusion, and how many rows are handed
    to the re-ranker; it defaults to four times k and at least 20, and is never less
    than k. Both stages work on rows that already passed the predicate.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if k <= 0:
        raise ValueError("k must be positive")
    if mode != "vector" and not store.term_index_ready():
        raise TermIndexMissing("this database has no keyword index; run: scoped-rag reindex")
    depth = max(candidates, k) if candidates else max(4 * k, 20)
    pool = depth if reranker is not None else k

    if mode == "vector":
        hits = search(store, embedder, asker=asker, query=query, k=pool)
    elif mode == "keyword":
        hits = keyword_search(store, embedder, asker=asker, query=query, k=pool)
    else:
        hits = fuse(
            search(store, embedder, asker=asker, query=query, k=depth),
            keyword_search(store, embedder, asker=asker, query=query, k=depth),
        )[:pool]

    if reranker is None or not hits:
        return hits[:k]
    scores = reranker.score(query, [h.text for h in hits])
    if len(scores) != len(hits):
        raise ValueError(f"re-ranker returned {len(scores)} scores for {len(hits)} chunks")
    ranked = sorted(
        (replace(h, rerank=float(s)) for h, s in zip(hits, scores)),
        key=lambda h: (-(h.rerank or 0.0), h.chunk_id),
    )
    return ranked[:k]
