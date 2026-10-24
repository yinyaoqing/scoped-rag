from __future__ import annotations

import pytest

from scoped_rag.answer.generate import (
    REFUSAL,
    Answer,
    CitationError,
    answer_question,
    build_prompt,
)
from scoped_rag.answer.providers import Completion, FakeProvider
from scoped_rag.embed.fake import FakeEmbedder
from scoped_rag.retrieve.search import Hit, search
from scoped_rag.store.repo import Store


def test_prompt_only_contains_permitted_chunks(store: Store, embedder: FakeEmbedder):
    hits = search(store, embedder, asker="u:alice", query="調薪級距", k=50)
    prompt = build_prompt("調薪級距是多少", hits)
    assert "salary" not in prompt
    assert "薪資調整級距" not in prompt


def test_refuses_when_nothing_is_readable(store: Store, embedder: FakeEmbedder):
    store.add_user("u:nobody", "Nobody")
    store.set_visibility("handbook", "restricted")
    result = answer_question(
        store, embedder, FakeProvider(), asker="u:nobody", question="育嬰假多久", k=10
    )
    assert result.refused
    assert result.text == REFUSAL
    assert result.retrieved == []


def test_refuses_when_everything_is_too_far(store: Store, embedder: FakeEmbedder):
    result = answer_question(
        store,
        embedder,
        FakeProvider(),
        asker="u:alice",
        question="育嬰假多久",
        k=10,
        max_distance=0.0,
    )
    assert result.refused
    assert result.retrieved, "the rows were readable; they were just not close enough"
    assert result.used == []


def test_answer_cites_a_chunk_it_was_given(store: Store, embedder: FakeEmbedder):
    result = answer_question(
        store,
        embedder,
        FakeProvider(),
        asker="u:alice",
        question="彈性上下班區間是多久",
        k=10,
        max_distance=1.2,
    )
    assert not result.refused
    assert result.citations
    assert set(result.citations) <= set(result.retrieved_chunk_ids)


def test_a_citation_outside_the_permitted_set_raises(store: Store, embedder: FakeEmbedder):
    class Liar:
        @property
        def name(self) -> str:
            return "liar"

        def complete(self, system: str, user: str, *, max_tokens: int = 700) -> Completion:
            return Completion("經理調薪區間是百分之三到八 [salary:0]")

    with pytest.raises(CitationError) as exc:
        answer_question(
            store,
            embedder,
            Liar(),
            asker="u:alice",
            question="調薪區間",
            k=10,
            max_distance=1.2,
        )
    assert "salary:0" in str(exc.value)


def test_cost_is_zero_for_the_fake_provider(store: Store, embedder: FakeEmbedder):
    result = answer_question(
        store, embedder, FakeProvider(), asker="u:alice", question="育嬰假", k=10, max_distance=1.2
    )
    assert result.cost_usd == 0.0


def test_claude_pricing_is_applied_per_token():
    c = Completion("x", input_tokens=1_000_000, output_tokens=1_000_000, model="claude-haiku-5-5")
    assert c.cost_usd() == pytest.approx(0.60)


def test_answer_defaults_are_empty():
    a = Answer("x")
    assert a.retrieved == [] and a.used == [] and a.citations == []


def test_hit_is_hashable_enough_for_set_maths():
    h = Hit("c1", "d1", "t", "body", 0.1)
    assert {h} == {Hit("c1", "d1", "t", "body", 0.1)}
