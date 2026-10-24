"""The privilege-escalation eval.

`leak_rate` is the number that matters: a single forbidden chunk reaching the
asker fails the run. `permitted_recall` is its counterweight, because a retriever
that returns nothing never leaks and is also useless. `refusal_correctness` checks
the answer path refuses instead of guessing when nothing readable is close.

Every case gets a freshly built store, so one case's mutations cannot colour the
next. The fake embedder and provider keep the whole thing deterministic and free:
what is under test is the access predicate, not an embedding model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .answer.generate import answer_question
from .answer.providers import FakeProvider
from .embed.fake import FakeEmbedder
from .ingest.chunk import chunk_text
from .store.repo import Store, seed_roles


@dataclass
class CaseResult:
    id: str
    family: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    cases: int = 0
    leak_rate: float = 0.0
    permitted_recall: float = 0.0
    refusal_correctness: float = 0.0
    passed: bool = False
    results: list[CaseResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "leak_rate": round(self.leak_rate, 4),
            "permitted_recall": round(self.permitted_recall, 4),
            "refusal_correctness": round(self.refusal_correctness, 4),
            "passed": self.passed,
            "failures": [
                {"id": r.id, "family": r.family, "detail": r.detail}
                for r in self.results
                if not r.passed
            ],
        }


def _build(spec: dict[str, Any], root: Path, embedder: FakeEmbedder) -> Store:
    store = Store.open("sqlite://:memory:", dim=embedder.dim)
    store.init_schema()
    seed_roles(store)
    for gid, name in (spec.get("groups") or {}).items():
        store.add_group(gid, name)
    for uid, info in (spec.get("users") or {}).items():
        store.add_user(uid, info.get("name", uid))
        for gid in info.get("groups") or []:
            store.add_member(uid, gid)
    for doc in spec.get("documents") or []:
        body = (root / doc["file"]).read_text(encoding="utf-8")
        store.add_document(
            doc["title"],
            source=doc["file"],
            visibility=doc.get("visibility", "restricted"),
            doc_id=doc["id"],
        )
        specs = chunk_text(body, size=120, overlap=24)
        store.add_chunks(
            doc["id"], specs, embedder.embed_documents([c.text for c in specs]), embedder.name
        )
        for principal in doc.get("grants") or []:
            store.grant(doc["id"], principal)
    store.commit()
    return store


def _mutate(store: Store, steps: list[dict[str, Any]]) -> None:
    for step in steps:
        op = step["op"]
        if op == "grant":
            store.grant(step["document"], step["principal"])
        elif op == "revoke":
            store.revoke(step["document"], step["principal"])
        elif op == "add_member":
            store.add_member(step["user"], step["group"])
        elif op == "remove_member":
            store.remove_member(step["user"], step["group"])
        elif op == "set_visibility":
            store.set_visibility(step["document"], step["visibility"])
        else:
            raise ValueError(f"unknown mutation op {op!r}")
    store.commit()


def run(cases_path: str | Path = "evals/leakage/cases.yaml") -> Report:
    path = Path(cases_path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent
    k = int(spec.get("k", 20))
    embedder = FakeEmbedder(dim=64)
    provider = FakeProvider()

    report = Report()
    forbidden_checks = forbidden_clean = 0
    expected_checks = expected_hit = 0
    refusal_checks = refusal_right = 0

    for case in spec["cases"]:
        store = _build(spec, root, embedder)
        if case.get("mutate"):
            _mutate(store, case["mutate"])

        result = answer_question(
            store,
            embedder,
            provider,
            asker=case["asker"],
            question=case["question"],
            k=k,
        )
        retrieved_docs = {h.document_id for h in result.retrieved}
        all_retrieved = set(result.retrieved_chunk_ids)
        problems: list[str] = []

        for doc in case.get("forbidden_docs") or []:
            forbidden_checks += 1
            leaked = set(store.chunk_ids(doc)) & all_retrieved
            if leaked:
                problems.append(f"leaked {doc}: {sorted(leaked)}")
            else:
                forbidden_clean += 1

        for doc in case.get("expect_docs") or []:
            expected_checks += 1
            if doc in retrieved_docs:
                expected_hit += 1
            else:
                problems.append(f"missed {doc}")

        if "expect_refusal" in case:
            refusal_checks += 1
            if bool(case["expect_refusal"]) == result.refused:
                refusal_right += 1
            else:
                problems.append(f"refused={result.refused}, expected {case['expect_refusal']}")

        for cited in result.citations:
            if cited not in all_retrieved:
                problems.append(f"citation {cited} outside the permitted set")

        report.results.append(
            CaseResult(case["id"], case["family"], not problems, "; ".join(problems))
        )
        store.close()

    report.cases = len(report.results)
    report.leak_rate = 0.0 if not forbidden_checks else 1 - forbidden_clean / forbidden_checks
    report.permitted_recall = 1.0 if not expected_checks else expected_hit / expected_checks
    report.refusal_correctness = 1.0 if not refusal_checks else refusal_right / refusal_checks
    report.passed = report.leak_rate == 0.0 and all(r.passed for r in report.results)
    return report


def write_report(report: Report, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
