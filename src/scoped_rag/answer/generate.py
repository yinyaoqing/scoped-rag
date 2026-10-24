"""Turn a question into an answer the asker is allowed to receive.

Two guards sit after retrieval. The model only ever sees permitted chunks, and
every citation it returns is checked back against that permitted set: a citation
to anything else means the prompt or the parsing is broken, so it raises instead of
being quietly dropped. When nothing permitted is close enough, the answer is a
refusal -- never a guess assembled from whatever was nearest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..embed.base import Embedder
from ..retrieve.search import Hit, search
from ..store.repo import Store
from .providers import _LABEL, LlmProvider

SYSTEM = (
    "你是企業文件問答助手。只能依據提供的文件片段回答，不得使用其他知識。"
    "每個陳述後面都要以 [片段編號] 標註來源。"
    "若提供的片段無法回答問題，直接說明文件中沒有相關內容，不要推測。"
)

REFUSAL = "找不到你有權限查看的相關文件。"

# Cosine distance above which a chunk is treated as unrelated. Tuned against the
# fake embedder in the eval corpus; a real embedder wants its own number, which is
# why it is a parameter rather than a constant buried in the call.
DEFAULT_MAX_DISTANCE = 0.85


class CitationError(RuntimeError):
    """The model cited a chunk that was not in the permitted context."""


@dataclass(frozen=True)
class Answer:
    """`retrieved` is everything the access predicate let through; `used` is the
    subset close enough to go into the prompt. Keeping them apart matters: the
    access-control guarantee is about `retrieved`, while the relevance threshold
    is a product setting that should never be able to disguise a leak."""

    text: str
    citations: list[str] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    used: list[Hit] = field(default_factory=list)
    refused: bool = False
    cost_usd: float = 0.0

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [h.chunk_id for h in self.retrieved]


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[{h.chunk_id}] {h.title}\n{h.text}")
    context = "\n\n".join(blocks)
    return f"文件片段：\n\n{context}\n\n問題：{question}"


def answer_question(
    store: Store,
    embedder: Embedder,
    provider: LlmProvider,
    *,
    asker: str,
    question: str,
    k: int = 5,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> Answer:
    retrieved = search(store, embedder, asker=asker, query=question, k=k)
    near = [h for h in retrieved if h.distance <= max_distance]
    if not near:
        return Answer(REFUSAL, retrieved=retrieved, refused=True)

    completion = provider.complete(SYSTEM, build_prompt(question, near))
    allowed = {h.chunk_id for h in near}
    cited = [c for c in dict.fromkeys(_LABEL.findall(completion.text)) if ":" in c]
    stray = [c for c in cited if c not in allowed]
    if stray:
        raise CitationError(
            f"answer cited chunks outside the permitted set: {stray} (allowed: {sorted(allowed)})"
        )
    return Answer(
        completion.text,
        citations=cited,
        retrieved=retrieved,
        used=near,
        refused=False,
        cost_usd=completion.cost_usd(),
    )
