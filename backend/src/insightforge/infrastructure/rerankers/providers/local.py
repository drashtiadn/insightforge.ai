"""Local CrossEncoder backend shared by Cross Encoder and BGE rerankers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit
from insightforge.infrastructure.rerankers.base import RerankerProvider
from insightforge.infrastructure.rerankers.http import apply_scores, require_query

logger = get_logger(__name__)

# Injected scorer: query + documents in → scores out (same order / length).
RerankFn = Callable[[str, Sequence[str]], Sequence[float]]


def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


class LocalReranker(RerankerProvider):
    """On-device CrossEncoder via sentence-transformers or an injected function.

    Inject ``score_fn`` in tests (and offline CI). Production:
    ``uv sync --extra local-embeddings``.
    """

    default_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        *,
        model: str | None = None,
        score_fn: RerankFn | None = None,
    ) -> None:
        self._model = (model or "").strip() or self.default_model
        self._score_fn = score_fn
        self._encoder: Any | None = None

    @property
    def available(self) -> bool:
        if self._score_fn is not None:
            return True
        return sentence_transformers_available()

    @property
    def model(self) -> str:
        return self._model

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        cleaned = require_query(query)
        if not hits:
            return []

        logger.info(
            "local rerank started provider=%s model=%s candidates=%d top_n=%s",
            self.name.value,
            self._model,
            len(hits),
            top_n,
            extra={
                "provider": self.name.value,
                "model": self._model,
                "candidates": len(hits),
            },
        )
        documents = [hit.text for hit in hits]
        scores = self._score(cleaned, documents)
        ranked = apply_scores(hits, scores, top_n=top_n)
        logger.info(
            "local rerank finished provider=%s count=%d",
            self.name.value,
            len(ranked),
            extra={"provider": self.name.value, "count": len(ranked)},
        )
        return ranked

    def _score(self, query: str, documents: Sequence[str]) -> list[float]:
        if self._score_fn is not None:
            raw = self._score_fn(query, documents)
            return [float(value) for value in raw]

        encoder = self._get_encoder()
        pairs = [(query, text if text.strip() else " ") for text in documents]
        try:
            raw = encoder.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.warning(
                "local rerank failed provider=%s model=%s error=%s",
                self.name.value,
                self._model,
                exc,
                extra={"provider": self.name.value, "model": self._model},
            )
            raise ExternalServiceError(
                f"local rerank failed: {exc}",
                details={"provider": self.name.value, "model": self._model},
            ) from exc
        return [float(value) for value in raw]

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ExternalServiceError(
                "sentence-transformers is not installed; "
                "run `uv sync --extra local-embeddings` or inject score_fn",
                details={"provider": self.name.value, "model": self._model},
            ) from exc

        logger.info(
            "loading local reranker model=%s provider=%s",
            self._model,
            self.name.value,
            extra={"provider": self.name.value, "model": self._model},
        )
        try:
            self._encoder = CrossEncoder(self._model)
        except Exception as exc:
            raise ExternalServiceError(
                f"failed to load local reranker model: {exc}",
                details={"provider": self.name.value, "model": self._model},
            ) from exc
        return self._encoder
