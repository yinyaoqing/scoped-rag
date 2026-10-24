"""The eval is part of the product, so it is tested like the rest of it."""

from __future__ import annotations

from pathlib import Path

from scoped_rag.evaluate import run, write_report

CASES = Path("evals/leakage/cases.yaml")


def test_the_shipped_cases_pass_with_no_leaks():
    report = run(CASES)
    assert report.leak_rate == 0.0
    assert report.permitted_recall == 1.0
    assert report.refusal_correctness == 1.0
    assert report.passed, report.as_dict()["failures"]


def test_every_family_is_covered():
    report = run(CASES)
    families = {r.family for r in report.results}
    assert families == {
        "must_not_retrieve",
        "must_retrieve",
        "boundary",
        "refuses_when_nothing_permitted",
    }


def test_a_broken_predicate_is_caught(monkeypatch):
    """Guard the guard: if the access predicate stops filtering, the eval must fail.

    Without this, a regression that returns everything to everyone would still show
    leak_rate 0 as long as nobody noticed the eval had stopped being meaningful.
    """
    import scoped_rag.retrieve.search as search_mod

    def permissive(d, asker, *, is_admin=False, doc_alias="d"):
        return "1 = 1", {}

    monkeypatch.setattr(search_mod, "readable_predicate", permissive)
    report = run(CASES)
    assert report.leak_rate > 0.0
    assert not report.passed


def test_report_round_trips_to_json(tmp_path):
    report = run(CASES)
    out = tmp_path / "report.json"
    write_report(report, out)
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cases"] == report.cases
    assert data["leak_rate"] == 0.0
