"""Local sentence-transformers embeddings: no API cost, no network at query time.

Imported lazily so the rest of the package -- and the whole test suite -- runs
without the extra installed.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "local embeddings need the 'local-embed' extra: uv sync --extra local-embed"
            ) from exc
        self._model_name = model
        self._model = SentenceTransformer(model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
