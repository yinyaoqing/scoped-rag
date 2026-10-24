"""Persistence. Knows nothing about who may read what: that predicate lives in
`retrieve.acl` so there is exactly one copy of it.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .dialect import POSTGRES, SQLITE, Dialect, cosine_distance
from .schema import ADMIN_GROUP, ddl

DEFAULT_DSN = "sqlite:///scoped_rag.sqlite3"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source: str | None
    owner_id: str | None
    visibility: str


@dataclass(frozen=True)
class ChunkSpec:
    ord: int
    text: str
    start_offset: int
    end_offset: int


def _sqlite_path(dsn: str) -> str:
    for prefix in ("sqlite:///", "sqlite://", "sqlite:"):
        if dsn.startswith(prefix):
            return dsn[len(prefix) :] or ":memory:"
    return dsn


class Store:
    def __init__(self, conn: Any, dialect: Dialect, dim: int) -> None:
        self.conn = conn
        self.d = dialect
        self.dim = dim

    @classmethod
    def open(cls, dsn: str | None = None, *, dim: int = 64) -> Store:
        dsn = dsn or os.environ.get("SCOPED_RAG_DSN") or DEFAULT_DSN
        if dsn.startswith("postgres"):
            try:
                import psycopg
            except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
                raise RuntimeError("PostgreSQL needs the 'pg' extra: uv sync --extra pg") from exc
            return cls(psycopg.connect(dsn), POSTGRES, dim)
        conn = sqlite3.connect(_sqlite_path(dsn))
        conn.create_function("cosine", 2, cosine_distance, deterministic=True)
        return cls(conn, SQLITE, dim)

    # --- plumbing ---------------------------------------------------------
    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple]:
        cur = self.conn.cursor()
        cur.execute(sql, params or {})
        rows = cur.fetchall() if cur.description else []
        cur.close()
        return [tuple(r) for r in rows]

    def init_schema(self) -> None:
        if self.d is POSTGRES:
            self.execute("CREATE EXTENSION IF NOT EXISTS vector")
        for statement in ddl(self.d, self.dim):
            self.execute(statement)
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- principals -------------------------------------------------------
    def add_principal(self, pid: str, kind: str, name: str) -> str:
        p = self.d.ph
        if self.d is SQLITE:
            sql = f"INSERT OR REPLACE INTO principals (id, kind, name) VALUES ({p('id')}, {p('kind')}, {p('name')})"
        else:
            sql = (
                f"INSERT INTO principals (id, kind, name) VALUES ({p('id')}, {p('kind')}, {p('name')}) "
                "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name"
            )
        self.execute(sql, {"id": pid, "kind": kind, "name": name})
        return pid

    def add_user(self, pid: str, name: str) -> str:
        return self.add_principal(pid, "user", name)

    def add_group(self, pid: str, name: str) -> str:
        return self.add_principal(pid, "group", name)

    def add_member(self, user_id: str, group_id: str) -> None:
        p = self.d.ph
        head = "INSERT OR IGNORE INTO" if self.d is SQLITE else "INSERT INTO"
        tail = "" if self.d is SQLITE else " ON CONFLICT DO NOTHING"
        self.execute(
            f"{head} memberships (user_id, group_id) VALUES ({p('u')}, {p('g')}){tail}",
            {"u": user_id, "g": group_id},
        )

    def remove_member(self, user_id: str, group_id: str) -> None:
        p = self.d.ph
        self.execute(
            f"DELETE FROM memberships WHERE user_id = {p('u')} AND group_id = {p('g')}",
            {"u": user_id, "g": group_id},
        )

    def is_admin(self, user_id: str) -> bool:
        p = self.d.ph
        rows = self.execute(
            f"SELECT 1 FROM memberships WHERE user_id = {p('u')} AND group_id = {p('g')}",
            {"u": user_id, "g": ADMIN_GROUP},
        )
        return bool(rows)

    def principals(self, kind: str | None = None) -> list[tuple[str, str, str]]:
        if kind:
            p = self.d.ph
            rows = self.execute(
                f"SELECT id, kind, name FROM principals WHERE kind = {p('k')} ORDER BY id",
                {"k": kind},
            )
        else:
            rows = self.execute("SELECT id, kind, name FROM principals ORDER BY kind, id")
        return [(r[0], r[1], r[2]) for r in rows]

    def groups_of(self, user_id: str) -> list[str]:
        p = self.d.ph
        rows = self.execute(
            f"SELECT group_id FROM memberships WHERE user_id = {p('u')} ORDER BY group_id",
            {"u": user_id},
        )
        return [r[0] for r in rows]

    # --- documents --------------------------------------------------------
    def add_document(
        self,
        title: str,
        *,
        source: str | None = None,
        owner_id: str | None = None,
        visibility: str = "restricted",
        doc_id: str | None = None,
    ) -> str:
        if visibility not in ("org", "restricted"):
            raise ValueError(f"visibility must be 'org' or 'restricted', got {visibility!r}")
        doc_id = doc_id or uuid.uuid4().hex[:12]
        p = self.d.ph
        self.execute(
            "INSERT INTO documents (id, title, source, owner_id, visibility) "
            f"VALUES ({p('id')}, {p('t')}, {p('s')}, {p('o')}, {p('v')})",
            {"id": doc_id, "t": title, "s": source, "o": owner_id, "v": visibility},
        )
        return doc_id

    def add_chunks(
        self,
        doc_id: str,
        chunks: Sequence[ChunkSpec],
        vectors: Sequence[Sequence[float]],
        embedder: str,
    ) -> list[str]:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        p = self.d.ph
        ids: list[str] = []
        for chunk, vec in zip(chunks, vectors):
            if len(vec) != self.dim:
                raise ValueError(
                    f"embedder {embedder!r} produced dim {len(vec)}, store expects {self.dim}"
                )
            cid = f"{doc_id}:{chunk.ord}"
            self.execute(
                "INSERT INTO chunks (id, document_id, ord, text, start_offset, end_offset) "
                f"VALUES ({p('id')}, {p('d')}, {p('o')}, {p('t')}, {p('s')}, {p('e')})",
                {
                    "id": cid,
                    "d": doc_id,
                    "o": chunk.ord,
                    "t": chunk.text,
                    "s": chunk.start_offset,
                    "e": chunk.end_offset,
                },
            )
            self.execute(
                "INSERT INTO embeddings (chunk_id, embedder, dim, vector) "
                f"VALUES ({p('id')}, {p('m')}, {p('n')}, {p('v')})",
                {"id": cid, "m": embedder, "n": len(vec), "v": self.d.encode(list(vec))},
            )
            ids.append(cid)
        return ids

    def grant(self, doc_id: str, principal_id: str) -> None:
        p = self.d.ph
        head = "INSERT OR IGNORE INTO" if self.d is SQLITE else "INSERT INTO"
        tail = "" if self.d is SQLITE else " ON CONFLICT DO NOTHING"
        self.execute(
            f"{head} grants (document_id, principal_id) VALUES ({p('d')}, {p('p')}){tail}",
            {"d": doc_id, "p": principal_id},
        )

    def revoke(self, doc_id: str, principal_id: str) -> None:
        p = self.d.ph
        self.execute(
            f"DELETE FROM grants WHERE document_id = {p('d')} AND principal_id = {p('p')}",
            {"d": doc_id, "p": principal_id},
        )

    def set_visibility(self, doc_id: str, visibility: str) -> None:
        if visibility not in ("org", "restricted"):
            raise ValueError(f"visibility must be 'org' or 'restricted', got {visibility!r}")
        p = self.d.ph
        self.execute(
            f"UPDATE documents SET visibility = {p('v')} WHERE id = {p('d')}",
            {"v": visibility, "d": doc_id},
        )

    def documents(self) -> list[Document]:
        rows = self.execute(
            "SELECT id, title, source, owner_id, visibility FROM documents ORDER BY title"
        )
        return [Document(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def grants_of(self, doc_id: str) -> list[str]:
        p = self.d.ph
        rows = self.execute(
            f"SELECT principal_id FROM grants WHERE document_id = {p('d')} ORDER BY principal_id",
            {"d": doc_id},
        )
        return [r[0] for r in rows]

    def chunk_ids(self, doc_id: str | None = None) -> list[str]:
        if doc_id is None:
            return [r[0] for r in self.execute("SELECT id FROM chunks ORDER BY id")]
        p = self.d.ph
        rows = self.execute(
            f"SELECT id FROM chunks WHERE document_id = {p('d')} ORDER BY ord", {"d": doc_id}
        )
        return [r[0] for r in rows]

    def chunk_text(self, chunk_id: str) -> str | None:
        p = self.d.ph
        rows = self.execute(f"SELECT text FROM chunks WHERE id = {p('c')}", {"c": chunk_id})
        return rows[0][0] if rows else None


def seed_roles(store: Store, departments: Iterable[str] = ()) -> None:
    """The three roles these enterprise knowledge products ship with, plus departments."""
    store.add_group(ADMIN_GROUP, "管理員")
    store.add_group("g:manager", "經理")
    store.add_group("g:employee", "員工")
    for dept in departments:
        store.add_group(f"g:{dept}", dept)
