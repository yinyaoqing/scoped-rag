"""Schema shared by both backends.

`start_offset` / `end_offset` rather than `start` / `end` because END is reserved
in PostgreSQL. Roles (admin, manager, employee) and departments are both rows in
`principals` with kind='group', so a grant can name either a person or a group
without a second code path.
"""

from __future__ import annotations

from .dialect import Dialect

ADMIN_GROUP = "g:admin"


def ddl(d: Dialect, dim: int) -> list[str]:
    ts = d.timestamp_default()
    vec = d.vector_column(dim)
    return [
        """CREATE TABLE IF NOT EXISTS principals (
             id TEXT PRIMARY KEY,
             kind TEXT NOT NULL CHECK (kind IN ('user', 'group')),
             name TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS memberships (
             user_id TEXT NOT NULL,
             group_id TEXT NOT NULL,
             PRIMARY KEY (user_id, group_id))""",
        f"""CREATE TABLE IF NOT EXISTS documents (
             id TEXT PRIMARY KEY,
             title TEXT NOT NULL,
             source TEXT,
             owner_id TEXT,
             visibility TEXT NOT NULL CHECK (visibility IN ('org', 'restricted')),
             created_at TIMESTAMP DEFAULT {ts})""",
        """CREATE TABLE IF NOT EXISTS grants (
             document_id TEXT NOT NULL,
             principal_id TEXT NOT NULL,
             PRIMARY KEY (document_id, principal_id))""",
        """CREATE TABLE IF NOT EXISTS chunks (
             id TEXT PRIMARY KEY,
             document_id TEXT NOT NULL,
             ord INTEGER NOT NULL,
             text TEXT NOT NULL,
             start_offset INTEGER NOT NULL,
             end_offset INTEGER NOT NULL)""",
        f"""CREATE TABLE IF NOT EXISTS embeddings (
             chunk_id TEXT PRIMARY KEY,
             embedder TEXT NOT NULL,
             dim INTEGER NOT NULL,
             vector {vec} NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (document_id)",
        "CREATE INDEX IF NOT EXISTS idx_grants_doc ON grants (document_id)",
        "CREATE INDEX IF NOT EXISTS idx_memberships_user ON memberships (user_id)",
    ]
