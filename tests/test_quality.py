"""The quality eval's labels and plumbing. Its numbers are not asserted here: they
depend on the models, and the fakes used in tests say nothing about quality."""

from __future__ import annotations

from pathlib import Path

from scoped_rag import quality
from scoped_rag.embed.fake import FakeEmbedder
from scoped_rag.evaluate import _build
from scoped_rag.rerank.fake import FakeReranker

SET = Path("evals/retrieval/set.yaml")


def test_every_label_points_at_a_chunk_the_asker_may_read():
    """A question whose answer the asker cannot reach, or whose `contains` string no
    chunk holds, would count as a miss for every configuration and quietly drag all
    the numbers down together."""
    spec = quality.load(SET)
    store = _build(
        spec, SET.parent, FakeEmbedder(64), size=spec["chunk_size"], overlap=spec["chunk_overlap"]
    )
    for q in spec["queries"]:
        assert q["doc"] in quality.readable_docs(spec, q["asker"]), q["id"]
        holders = [c for c in store.chunk_ids(q["doc"]) if q["contains"] in store.chunk_text(c)]
        assert holders, q["id"]


def test_every_query_type_is_present():
    spec = quality.load(SET)
    assert {q["type"] for q in spec["queries"]} == {"identifier", "paraphrase", "plain", "scoped"}
    assert len({q["id"] for q in spec["queries"]}) == len(spec["queries"])


def test_all_configurations_run_and_none_leaks():
    report = quality.run(SET, embedder=FakeEmbedder(64), reranker=FakeReranker())
    assert list(report["configs"]) == [label for label, _, _ in quality.CONFIGS]
    for label, config in report["configs"].items():
        assert config["leaks"] == [], label
        assert config["overall"]["n"] == report["queries"]
        assert 0.0 <= config["overall"]["hit@1"] <= config["overall"]["hit@3"] <= 1.0


def test_rerank_configurations_are_skipped_without_a_reranker():
    report = quality.run(SET, embedder=FakeEmbedder(64))
    assert list(report["configs"]) == ["vector", "keyword", "hybrid"]


def test_the_leak_check_is_independent_of_the_predicate(monkeypatch):
    """Break the SQL predicate; the eval must notice from the spec's own rules."""
    import scoped_rag.retrieve.search as search_mod

    monkeypatch.setattr(search_mod, "readable_predicate", lambda d, asker, **kw: ("1 = 1", {}))
    report = quality.run(SET, embedder=FakeEmbedder(64))
    assert all(config["leaks"] for config in report["configs"].values())


def test_report_table_and_json(tmp_path):
    report = quality.run(SET, embedder=FakeEmbedder(64), reranker=FakeReranker())
    table = quality.format_table(report)
    assert "hybrid+rerank" in table and "identifier hit@1" in table
    out = tmp_path / "r.json"
    quality.write_report(report, out)
    assert out.read_text(encoding="utf-8").startswith("{")
