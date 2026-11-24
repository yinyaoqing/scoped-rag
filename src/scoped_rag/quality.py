"""The retrieval-quality eval.

The leakage eval answers "did anything forbidden come back" and runs on fake
models, because that property must not depend on the model. This eval answers the
other question -- "did the right chunk come back, and how high" -- which depends on
the models entirely, so the numbers that matter come from a run with the real
embedder and the real re-ranker. A run with the fakes is still useful as a smoke
test of the plumbing and is what the test suite does.

Each question is run through every retrieval configuration against one store.
Metrics, all over the top `k` chunks:

- `hit@1`, `hit@3`, `hit@k`: a relevant chunk is first / in the first three / present
  at all. `hit@k` of a first stage is the ceiling for any re-ranker placed after it.
- `mrr`: mean reciprocal rank of the first relevant chunk (0 when absent).

Every result list is also checked against the spec's own access rules, computed
here without the SQL predicate, so a configuration that leaked would be reported
by something other than the code under test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from .embed.base import Embedder
from .evaluate import _build
from .rerank.base import Reranker
from .retrieve.search import retrieve
from .store.schema import ADMIN_GROUP

DEFAULT_SET = Path("evals/retrieval/set.yaml")

# (label, mode, uses the re-ranker)
CONFIGS = [
    ("vector", "vector", False),
    ("keyword", "keyword", False),
    ("hybrid", "hybrid", False),
    ("vector+rerank", "vector", True),
    ("hybrid+rerank", "hybrid", True),
]


def load(path: str | Path = DEFAULT_SET) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def readable_docs(spec: dict[str, Any], asker: str) -> set[str]:
    """Who may read what, straight from the spec."""
    groups = set((spec["users"].get(asker) or {}).get("groups") or [])
    if ADMIN_GROUP in groups:
        return {d["id"] for d in spec["documents"]}
    return {
        d["id"]
        for d in spec["documents"]
        if d.get("visibility", "restricted") == "org"
        or (set(d.get("grants") or []) & (groups | {asker}))
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@k": 0.0, "mrr": 0.0}
    ranks = [r["rank"] for r in rows]
    return {
        "n": n,
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "hit@3": round(sum(1 for r in ranks if r and r <= 3) / n, 4),
        "hit@k": round(sum(1 for r in ranks if r) / n, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
    }


def run(
    set_path: str | Path = DEFAULT_SET,
    *,
    embedder: Embedder,
    reranker: Reranker | None = None,
) -> dict[str, Any]:
    path = Path(set_path)
    spec = load(path)
    k = int(spec.get("k", 10))
    pool = int(spec.get("rerank_pool", 0)) or None
    store = _build(
        spec,
        path.parent,
        embedder,
        size=int(spec.get("chunk_size", 160)),
        overlap=int(spec.get("chunk_overlap", 30)),
    )
    doc_of = {cid: cid.rsplit(":", 1)[0] for cid in store.chunk_ids()}
    types = sorted({q["type"] for q in spec["queries"]})

    report: dict[str, Any] = {
        "set": path.as_posix(),
        "embedder": embedder.name,
        "reranker": reranker.name if reranker else None,
        "documents": len(spec["documents"]),
        "chunks": len(doc_of),
        "queries": len(spec["queries"]),
        "k": k,
        "rerank_pool": pool,
        "configs": {},
    }
    for label, mode, use_reranker in CONFIGS:
        if use_reranker and reranker is None:
            continue
        rows: list[dict[str, Any]] = []
        leaks: list[str] = []
        started = time.perf_counter()
        for q in spec["queries"]:
            hits = retrieve(
                store,
                embedder,
                asker=q["asker"],
                query=q["question"],
                k=k,
                mode=mode,
                reranker=reranker if use_reranker else None,
                candidates=pool,
            )
            allowed = readable_docs(spec, q["asker"])
            leaks += [f"{q['id']}: {h.chunk_id}" for h in hits if h.document_id not in allowed]
            rank = next(
                (
                    i
                    for i, h in enumerate(hits, start=1)
                    if h.document_id == q["doc"] and q["contains"] in h.text
                ),
                None,
            )
            rows.append({"id": q["id"], "type": q["type"], "rank": rank})
        elapsed = time.perf_counter() - started
        report["configs"][label] = {
            "overall": _summary(rows),
            "by_type": {t: _summary([r for r in rows if r["type"] == t]) for t in types},
            "misses": [r["id"] for r in rows if r["rank"] is None],
            "leaks": leaks,
            "ms_per_query": round(1000 * elapsed / max(1, len(rows)), 1),
        }
    store.close()
    return report


def format_table(report: dict[str, Any]) -> str:
    types = sorted(next(iter(report["configs"].values()))["by_type"]) if report["configs"] else []
    k = report["k"]
    head = ["config", "hit@1", "hit@3", f"hit@{k}", "mrr"]
    head += [f"{t} hit@1" for t in types] + ["ms/q", "leaks"]
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for label, c in report["configs"].items():
        o = c["overall"]
        cells = [label, f"{o['hit@1']:.3f}", f"{o['hit@3']:.3f}", f"{o['hit@k']:.3f}"]
        cells.append(f"{o['mrr']:.3f}")
        cells += [f"{c['by_type'][t]['hit@1']:.3f}" for t in types]
        cells += [f"{c['ms_per_query']:.1f}", str(len(c["leaks"]))]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def write_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
