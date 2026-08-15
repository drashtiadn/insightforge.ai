"""Local embedding provider via sentence-transformers (optional dependency)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import EmbeddingResult
from insightforge.infrastructure.embeddings.base import EmbeddingProvider
from insightforge.infrastructure.embeddings.http import batched, require_texts
from insightforge.shared.enums import EmbeddingInputType, EmbeddingProviderHint

logger = get_logger(__name__)

# Injected encoder: texts in → vectors out (same order / length).
LocalEmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


class LocalEmbeddingProvider(EmbeddingProvider):
    """On-device embeddings via sentence-transformers or an injected function.

    Inject ``embed_fn`` in tests (and offline CI) so the heavy model package is
    not required. Production installs: ``uv sync --extra local-embeddings``.
    """

    name = EmbeddingProviderHint.LOCAL

    def __init__(
        self,
        *,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_fn: LocalEmbedFn | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model = (model or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"
        self._embed_fn = embed_fn
        self._batch_size = max(1, batch_size)
        self._encoder: Any | None = None

    @property
    def available(self) -> bool:
        if self._embed_fn is not None:
            return True
        return _sentence_transformers_available()

    @property
    def model(self) -> str:
        return self._model

    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> EmbeddingResult:
        cleaned = require_texts(texts)
        if not cleaned:
            return EmbeddingResult(
                vectors=[],
                model=self._model,
                provider=self.name,
                dimensions=0,
                input_type=input_type,
                usage_tokens=0,
                metadata={"backend": "local"},
            )

        logger.info(
            "local embed started count=%d model=%s input_type=%s batch_size=%d",
            len(cleaned),
            self._model,
            input_type.value,
            self._batch_size,
            extra={
                "provider": self.name.value,
                "model": self._model,
                "count": len(cleaned),
                "input_type": input_type.value,
            },
        )

        vectors: list[list[float]] = []
        for batch in batched(cleaned, self._batch_size):
            vectors.extend(self._encode_batch(batch))

        if len(vectors) != len(cleaned):
            raise ExternalServiceError(
                "local embeddings returned unexpected count",
                details={
                    "provider": self.name.value,
                    "expected": len(cleaned),
                    "actual": len(vectors),
                },
            )

        dimensions = len(vectors[0]) if vectors else 0
        logger.info(
            "local embed finished count=%d dims=%d",
            len(vectors),
            dimensions,
            extra={
                "provider": self.name.value,
                "count": len(vectors),
                "dimensions": dimensions,
            },
        )
        return EmbeddingResult(
            vectors=vectors,
            model=self._model,
            provider=self.name,
            dimensions=dimensions,
            input_type=input_type,
            metadata={"backend": "local"},
        )

    def _encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            raw = self._embed_fn(texts)
            return [list(vector) for vector in raw]

        encoder = self._get_encoder()
        try:
            # normalize_embeddings keeps vectors unit-length for cosine search.
            matrix = encoder.encode(
                list(texts),
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as exc:
            logger.warning(
                "local embed failed model=%s error=%s",
                self._model,
                exc,
                extra={"provider": self.name.value, "model": self._model},
            )
            raise ExternalServiceError(
                f"local embeddings failed: {exc}",
                details={"provider": self.name.value, "model": self._model},
            ) from exc

        return [[float(value) for value in row] for row in matrix]

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ExternalServiceError(
                "sentence-transformers is not installed; "
                "run `uv sync --extra local-embeddings` or inject embed_fn",
                details={"provider": self.name.value, "model": self._model},
            ) from exc

        logger.info(
            "loading local embedding model model=%s",
            self._model,
            extra={"provider": self.name.value, "model": self._model},
        )
        try:
            self._encoder = SentenceTransformer(self._model)
        except Exception as exc:
            raise ExternalServiceError(
                f"failed to load local embedding model: {exc}",
                details={"provider": self.name.value, "model": self._model},
            ) from exc
        return self._encoder
