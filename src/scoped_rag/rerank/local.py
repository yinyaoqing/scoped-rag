"""A local cross-encoder re-ranker: no API cost, no network at query time.

Imported lazily, like the local embedder, so the package and the test suite run
without the extra installed.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MODEL = "BAAI/bge-reranker-base"


class LocalReranker:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "local re-ranking needs the 'local-embed' extra: uv sync --extra local-embed"
            ) from exc
        self._model_name = model
        self._model = CrossEncoder(model)

    @property
    def name(self) -> str:
        return self._model_name

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]
