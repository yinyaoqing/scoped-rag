"""Permission-scoped vector search.

The point of this file is that the access predicate and the vector ordering go to
the database together: WHERE filters, ORDER BY ranks what survived, LIMIT takes k.
Filtering afterwards would both leak (result counts reveal documents you cannot
read) and rank worse (your top k gets eaten by rows you never get to see).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..embed.base import Embedder
from ..store.repo import Store
from .acl import readable_predicate


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document_id: str
    title: str
    text: str
    distance: float


def search(
    store: Store,
    embedder: Embedder,
    *,
    asker: str,
    query: str,
    k: int = 5,
) -> list[Hit]:
    if k <= 0:
        raise ValueError("k must be positive")
    acl_sql, acl_params = readable_predicate(
        store.d, asker, is_admin=store.is_admin(asker), doc_alias="d"
    )
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
