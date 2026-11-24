"""The keyword leg, fusion and re-ranking, held to the same rule as the vector leg:
nothing the asker cannot read may influence what they get back."""

from __future__ import annotations

import pytest

from scoped_rag.answer.generate import answer_question
from scoped_rag.answer.providers import FakeProvider
from scoped_rag.embed.fake import FakeEmbedder
from scoped_rag.ingest.chunk import chunk_text
from scoped_rag.rerank.fake import FakeReranker
from scoped_rag.retrieve.search import Hit, TermIndexMissing, fuse, keyword_search, retrieve
from scoped_rag.store.repo import Store
from scoped_rag.terms import terms


def docs_of(hits) -> set[str]:
    return {h.document_id for h in hits}


def add_doc(store: Store, embedder: FakeEmbedder, doc_id: str, body: str, grants=()):
    store.add_document(doc_id, visibility="restricted", doc_id=doc_id)
    specs = chunk_text(body, size=200, overlap=0)
    store.add_chunks(doc_id, specs, embedder.embed_documents([c.text for c in specs]), "fake-64")
    for g in grants:
        store.grant(doc_id, g)
    store.commit()


# --- terms ----------------------------------------------------------------


def test_identifiers_are_kept_whole_and_in_parts():
    assert terms("表單 FIN-204") == ["表單", "fin-204", "fin", "204"]


def test_cjk_runs_become_bigrams_and_a_lone_character_survives():
    assert terms("育嬰假") == ["育嬰", "嬰假"]
    assert terms("假") == ["假"]


def test_text_with_no_terms_matches_nothing(store: Store, embedder: FakeEmbedder):
    assert keyword_search(store, embedder, asker="u:root", query="？！…", k=5) == []


# --- the keyword leg is scoped ---------------------------------------------


def test_keyword_leg_finds_an_exact_identifier(store: Store, embedder: FakeEmbedder):
    add_doc(store, embedder, "forms", "費用報銷請填寫表單 FIN-204。", ["g:employee"])
    add_doc(store, embedder, "other", "出差申請請填寫表單 FIN-240。", ["g:employee"])
    hits = keyword_search(store, embedder, asker="u:alice", query="FIN-204", k=5)
    assert hits[0].document_id == "forms"
    assert hits[0].keyword is not None and hits[0].keyword > 0


@pytest.mark.parametrize("mode", ["keyword", "hybrid"])
def test_employee_cannot_reach_the_manager_document(store, embedder, mode):
    hits = retrieve(store, embedder, asker="u:alice", query="年度調薪區間", k=10, mode=mode)
    assert "salary" not in docs_of(hits)
    hits = retrieve(store, embedder, asker="u:bob", query="年度調薪區間", k=10, mode=mode)
    assert "salary" in docs_of(hits)


def test_hidden_documents_do_not_move_visible_scores(store: Store, embedder: FakeEmbedder):
    """BM25 statistics come from the readable set, so they cannot be used as an oracle.

    With corpus-wide document frequency, adding a manager-only document full of the
    query's terms would lower their weight and change Alice's scores -- she could
    learn that such a document exists, and by probing terms, what it says.
    """
    query = "上班時間 育嬰留職停薪"
    before = keyword_search(store, embedder, asker="u:alice", query=query, k=10)
    assert before

    add_doc(
        store, embedder, "secret", "上班時間調整與育嬰留職停薪主管處理原則。" * 5, ["g:manager"]
    )
    after = keyword_search(store, embedder, asker="u:alice", query=query, k=10)

    assert [(h.chunk_id, h.keyword) for h in after] == [(h.chunk_id, h.keyword) for h in before]
    # The same document does change the picture for someone who may read it.
    bob = keyword_search(store, embedder, asker="u:bob", query=query, k=10)
    assert "secret" in docs_of(bob)


def test_filter_runs_before_keyword_ranking(store: Store, embedder: FakeEmbedder):
    """k=1 and the best keyword match is forbidden: Alice still gets her own best row."""
    query = "經理級年度調薪區間為百分之三到百分之八"
    assert keyword_search(store, embedder, asker="u:root", query=query, k=1)[0].document_id == (
        "salary"
    )
    alice = keyword_search(store, embedder, asker="u:alice", query=query, k=1)
    assert len(alice) == 1 and alice[0].document_id != "salary"


# --- fusion ----------------------------------------------------------------


def hit(cid: str, **kw) -> Hit:
    return Hit(cid, cid.split(":")[0], "t", "x", 0.5, **kw)


def test_fusion_rewards_agreement_between_the_legs():
    vector = [hit("a:0"), hit("b:0"), hit("c:0")]
    keyword = [hit("c:0", keyword=3.0), hit("b:0", keyword=2.0), hit("d:0", keyword=1.0)]
    fused = fuse(vector, keyword)
    # b and c appear in both lists and beat a and d, which each lead only one.
    assert [h.chunk_id for h in fused] == ["c:0", "b:0", "a:0", "d:0"]
    assert {h.chunk_id for h in fused} == {"a:0", "b:0", "c:0", "d:0"}
    by_id = {h.chunk_id: h for h in fused}
    assert by_id["c:0"].keyword == 3.0, "the keyword score survives the merge"
    assert by_id["a:0"].keyword is None
    assert all(h.fused is not None for h in fused)


def test_fusion_adds_nothing_the_legs_did_not_return(store: Store, embedder: FakeEmbedder):
    q = "調薪 育嬰假 上班時間"
    hybrid = retrieve(store, embedder, asker="u:alice", query=q, k=50, mode="hybrid")
    readable = set(store.chunk_ids("handbook")) | set(store.chunk_ids("leave"))
    assert {h.chunk_id for h in hybrid} <= readable


# --- re-ranking ------------------------------------------------------------


def test_reranker_reorders_but_never_adds(store: Store, embedder: FakeEmbedder):
    q = "彈性上下班區間"
    plain = retrieve(store, embedder, asker="u:alice", query=q, k=50, mode="hybrid")
    ranked = retrieve(
        store, embedder, asker="u:alice", query=q, k=50, mode="hybrid", reranker=FakeReranker()
    )
    assert {h.chunk_id for h in ranked} == {h.chunk_id for h in plain}
    assert all(h.rerank is not None for h in ranked)
    scores = [h.rerank for h in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0].document_id == "handbook"


def test_reranker_only_sees_permitted_text(store: Store, embedder: FakeEmbedder):
    seen: list[str] = []

    class Spy(FakeReranker):
        def score(self, query, texts):
            seen.extend(texts)
            return super().score(query, texts)

    retrieve(store, embedder, asker="u:alice", query="年度調薪區間", k=5, reranker=Spy())
    assert seen
    salary = {store.chunk_text(c) for c in store.chunk_ids("salary")}
    assert not salary & set(seen)


def test_a_reranker_that_miscounts_is_an_error(store: Store, embedder: FakeEmbedder):
    class Broken(FakeReranker):
        def score(self, query, texts):
            return [1.0]

    with pytest.raises(ValueError, match="re-ranker returned"):
        retrieve(store, embedder, asker="u:root", query="調薪 育嬰假", k=5, reranker=Broken())


def test_unknown_mode_is_an_error(store: Store, embedder: FakeEmbedder):
    with pytest.raises(ValueError, match="mode must be one of"):
        retrieve(store, embedder, asker="u:root", query="調薪", mode="bm42")


# --- the answer path -------------------------------------------------------


def test_rerank_score_can_be_the_relevance_gate(store: Store, embedder: FakeEmbedder):
    kwargs = {"asker": "u:alice", "k": 10, "mode": "hybrid", "reranker": FakeReranker()}
    on_topic = answer_question(
        store, embedder, FakeProvider(), question="彈性上下班區間", min_rerank=0.5, **kwargs
    )
    assert not on_topic.refused
    assert all(h.rerank >= 0.5 for h in on_topic.used)

    off_topic = answer_question(
        store, embedder, FakeProvider(), question="量子退相干時間", min_rerank=0.5, **kwargs
    )
    assert off_topic.refused
    assert off_topic.retrieved, "refusing is about relevance; the permitted set is still reported"


# --- an older database ------------------------------------------------------


def test_a_database_without_a_term_index_fails_loudly(store: Store, embedder: FakeEmbedder):
    """Otherwise `--mode hybrid` on an older database would be vector search in disguise."""
    for wipe in ("DELETE FROM chunk_terms", "DROP TABLE chunk_terms"):
        store.execute(wipe)
        for mode in ("keyword", "hybrid"):
            with pytest.raises(TermIndexMissing, match="scoped-rag reindex"):
                retrieve(store, embedder, asker="u:root", query="育嬰假", k=5, mode=mode)
        assert retrieve(store, embedder, asker="u:root", query="育嬰假", k=5), "vector still works"
    store.init_schema()
    store.reindex_terms()
    assert retrieve(store, embedder, asker="u:root", query="育嬰假", k=5, mode="hybrid")


def test_reindex_rebuilds_the_term_index(store: Store, embedder: FakeEmbedder):
    store.execute("DELETE FROM chunk_terms")
    assert keyword_search(store, embedder, asker="u:root", query="育嬰假", k=5) == []
    assert store.reindex_terms() == len(store.chunk_ids())
    assert keyword_search(store, embedder, asker="u:root", query="育嬰假", k=5)
