"""What differs between SQLite and PostgreSQL, and nothing more.

The access-control predicate, the joins, the ordering and the LIMIT are identical
SQL on both backends. Only three things change: how a parameter is spelled, how a
vector is stored and bound, and the expression that measures distance. Keeping the
difference this small is the point -- the permission logic exists once, so the tests
that prove it does not leak cover the production backend too.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any


def cosine_distance(a: bytes, b: bytes) -> float:
    """Registered into SQLite so it can ORDER BY distance like pgvector's `<=>`."""
    va = struct.unpack(f"<{len(a) // 4}f", a)
    vb = struct.unpack(f"<{len(b) // 4}f", b)
    if len(va) != len(vb):
        return 2.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0.0 or nb == 0.0:
        return 2.0
    return 1.0 - dot / (na * nb)


@dataclass(frozen=True)
class Dialect:
    name: str

    def ph(self, key: str) -> str:
        """Named-parameter placeholder."""
        return f":{key}" if self.name == "sqlite" else f"%({key})s"

    def vector_column(self, dim: int) -> str:
        return "BLOB" if self.name == "sqlite" else f"vector({dim})"

    def timestamp_default(self) -> str:
        return "CURRENT_TIMESTAMP" if self.name == "sqlite" else "now()"

    def encode(self, vec: list[float]) -> Any:
        if self.name == "sqlite":
            return struct.pack(f"<{len(vec)}f", *vec)
        return "[" + ",".join(repr(float(x)) for x in vec) + "]"

    def distance(self, column: str, key: str) -> str:
        if self.name == "sqlite":
            return f"cosine({column}, {self.ph(key)})"
        return f"({column} <=> {self.ph(key)}::vector)"


SQLITE = Dialect("sqlite")
POSTGRES = Dialect("postgres")
