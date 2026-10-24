"""The access-control predicate. One copy, used by every query that reads chunks.

A document is readable when it is org-wide, or granted to the asker directly, or
granted to a group the asker belongs to. Admins read everything, stated here as an
explicit branch rather than buried in a role table, because a silent "admin sees
all" is exactly the rule that should be easy to find.
"""

from __future__ import annotations

from ..store.dialect import Dialect


def readable_predicate(
    d: Dialect, asker: str, *, is_admin: bool = False, doc_alias: str = "d"
) -> tuple[str, dict[str, str]]:
    """Return a SQL fragment and its parameters. The fragment is backend-independent."""
    if is_admin:
        return "1 = 1", {}
    who = d.ph("asker")
    sql = (
        f"({doc_alias}.visibility = 'org'"
        f" OR EXISTS (SELECT 1 FROM grants g"
        f"            WHERE g.document_id = {doc_alias}.id AND g.principal_id = {who})"
        f" OR EXISTS (SELECT 1 FROM grants g"
        f"            JOIN memberships m ON m.group_id = g.principal_id"
        f"            WHERE g.document_id = {doc_alias}.id AND m.user_id = {who}))"
    )
    return sql, {"asker": asker}
