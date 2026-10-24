"""What this project claims, stated as tests.

The central one is `test_filter_runs_before_ranking`: it fails for any
implementation that retrieves first and filters afterwards.
"""

from __future__ import annotations

from scoped_rag.embed.fake import FakeEmbedder
from scoped_rag.retrieve.search import search
from scoped_rag.store.repo import Store


def docs_of(hits) -> set[str]:
    return {h.document_id for h in hits}


def test_employee_cannot_reach_the_manager_document(store: Store, embedder: FakeEmbedder):
    hits = search(store, embedder, asker="u:alice", query="年度調薪區間是多少", k=10)
    assert "salary" not in docs_of(hits)


def test_manager_can_reach_it(store: Store, embedder: FakeEmbedder):
    hits = search(store, embedder, asker="u:bob", query="年度調薪區間是多少", k=10)
    assert "salary" in docs_of(hits)


def test_org_wide_document_reaches_everyone(store: Store, embedder: FakeEmbedder):
    for who in ("u:alice", "u:bob", "u:root"):
        hits = search(store, embedder, asker=who, query="彈性上下班區間", k=10)
        assert "handbook" in docs_of(hits), who


def test_admin_reaches_everything(store: Store, embedder: FakeEmbedder):
    hits = search(store, embedder, asker="u:root", query="調薪 育嬰假 上班時間", k=20)
    assert docs_of(hits) == {"handbook", "leave", "salary"}


def test_revoking_a_grant_removes_access(store: Store, embedder: FakeEmbedder):
    assert "leave" in docs_of(search(store, embedder, asker="u:alice", query="育嬰假", k=10))
    store.revoke("leave", "g:employee")
    assert "leave" not in docs_of(search(store, embedder, asker="u:alice", query="育嬰假", k=10))


def test_leaving_a_group_removes_access(store: Store, embedder: FakeEmbedder):
    assert "salary" in docs_of(search(store, embedder, asker="u:bob", query="調薪", k=10))
    store.remove_member("u:bob", "g:manager")
    assert "salary" not in docs_of(search(store, embedder, asker="u:bob", query="調薪", k=10))


def test_a_direct_grant_to_one_person_is_enough(store: Store, embedder: FakeEmbedder):
    store.grant("salary", "u:alice")
    assert "salary" in docs_of(search(store, embedder, asker="u:alice", query="調薪", k=10))


def test_filter_runs_before_ranking(store: Store, embedder: FakeEmbedder):
    """With k=1, a forbidden nearest neighbour must not cost the asker their own best hit.

    `query` is lifted from the salary document, so for an unrestricted search the
    top hit is a chunk Alice may not read. Filtering after ranking would hand her
    an empty result; filtering inside the query gives her the best chunk she is
    allowed to see.
    """
    query = "經理級年度調薪區間為百分之三到百分之八"
    everyone = search(store, embedder, asker="u:root", query=query, k=1)
    assert everyone and everyone[0].document_id == "salary"

    alice = search(store, embedder, asker="u:alice", query=query, k=1)
    assert len(alice) == 1
    assert alice[0].document_id != "salary"


def test_result_count_does_not_reveal_hidden_documents(store: Store, embedder: FakeEmbedder):
    """Alice asking a question only the salary document answers gets her own corpus,
    never a shorter list that hints something was removed."""
    alice = search(store, embedder, asker="u:alice", query="調薪級距", k=50)
    readable = set(store.chunk_ids("handbook")) | set(store.chunk_ids("leave"))
    assert {h.chunk_id for h in alice} == readable


def test_vectors_from_another_embedder_are_ignored(store: Store, embedder: FakeEmbedder):
    other = FakeEmbedder(dim=64)
    hits = search(store, other, asker="u:root", query="育嬰假", k=10)
    assert hits, "same name and dim, so the rows still match"

    class Renamed(FakeEmbedder):
        @property
        def name(self) -> str:
            return "fake-other"

    assert search(store, Renamed(dim=64), asker="u:root", query="育嬰假", k=10) == []
