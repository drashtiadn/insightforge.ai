"""Embedding service — select provider, embed documents/queries, adapt for chunking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Self

import httpx

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import EmbeddingResult
from insightforge.infrastructure.embeddings.base import EmbeddingProvider
from insightforge.infrastructure.embeddings.http import close_http_client, create_http_client
from insightforge.infrastructure.embeddings.providers import (
    LocalEmbeddingProvider,
    LocalEmbedFn,
    VoyageEmbeddingProvider,
)
from insightforge.shared.enums import EmbeddingInputType, EmbeddingProviderHint

logger = get_logger(__name__)

# Compatible with document.chunking.semantic.EmbedFn without a cross-import.
EmbedAdapterFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def parse_embedding_provider(value: str) -> EmbeddingProviderHint:
    """Parse settings/env value into ``EmbeddingProviderHint``."""

    cleaned = value.strip().lower()
    try:
        return EmbeddingProviderHint(cleaned)
    except ValueError as exc:
        allowed = ", ".join(hint.value for hint in EmbeddingProviderHint)
        raise ValidationFailedError(
            f"unknown embedding provider {value!r}; expected one of: {allowed}",
            details={"field": "embedding_provider", "value": value},
        ) from exc


def build_provider(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    local_embed_fn: LocalEmbedFn | None = None,
) -> EmbeddingProvider:
    """Construct the configured embedding provider."""

    hint = parse_embedding_provider(settings.embedding_provider)
    if hint is EmbeddingProviderHint.VOYAGE:
        if client is None:
            raise ValidationFailedError(
                "voyage embeddings require an httpx client",
                details={"field": "client"},
            )
        api_key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else None
        return VoyageEmbeddingProvider(
            client,
            api_key=api_key,
            model=settings.embedding_voyage_model,
            dimensions=settings.embedding_voyage_dimensions,
            batch_size=settings.embedding_batch_size,
        )

    return LocalEmbeddingProvider(
        model=settings.embedding_local_model,
        embed_fn=local_embed_fn,
        batch_size=settings.embedding_batch_size,
    )


class EmbeddingService:
    """Facade over a single embedding provider.

    When constructed via ``create_embedding_service`` for Voyage without an
    injected client, this service owns the shared ``httpx.Client`` and closes
    it on ``close()`` or context-manager exit.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        client: httpx.Client | None = None,
        owns_client: bool = False,
    ) -> None:
        self._provider = provider
        self._client = client
        self._owns_client = owns_client and client is not None

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def available(self) -> bool:
        return self._provider.available

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def client(self) -> httpx.Client | None:
        return self._client

    @property
    def owns_client(self) -> bool:
        return self._owns_client

    def close(self) -> None:
        """Close the owned HTTP client, if any. Idempotent."""

        if not self._owns_client:
            return
        close_http_client(self._client)
        self._client = None
        self._owns_client = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> EmbeddingResult:
        """Embed ``texts`` with the configured provider."""

        if not self._provider.available:
            raise ExternalServiceError(
                f"embedding provider unavailable: {self._provider.name.value}",
                details={"provider": self._provider.name.value},
            )

        logger.info(
            "embed started provider=%s model=%s count=%d input_type=%s",
            self._provider.name.value,
            self._provider.model,
            len(texts),
            input_type.value,
            extra={
                "provider": self._provider.name.value,
                "model": self._provider.model,
                "count": len(texts),
                "input_type": input_type.value,
            },
        )
        result = self._provider.embed(texts, input_type=input_type)
        logger.info(
            "embed finished provider=%s count=%d dims=%d",
            result.provider.value,
            len(result.vectors),
            result.dimensions,
            extra={
                "provider": result.provider.value,
                "count": len(result.vectors),
                "dimensions": result.dimensions,
            },
        )
        return result

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed corpus/document texts."""

        return self.embed(texts, input_type=EmbeddingInputType.DOCUMENT)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single retrieval query; returns one vector."""

        result = self.embed([text], input_type=EmbeddingInputType.QUERY)
        if not result.vectors:
            raise ExternalServiceError(
                "embedding provider returned no query vector",
                details={"provider": self._provider.name.value},
            )
        return result.vectors[0]

    def as_embed_fn(self) -> EmbedAdapterFn:
        """Adapter for semantic chunking (document input type)."""

        def _embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return self.embed_documents(texts).vectors

        return _embed


def create_embedding_service(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
    local_embed_fn: LocalEmbedFn | None = None,
) -> EmbeddingService:
    """Factory used by application code and tests.

    When the configured provider is Voyage and ``client`` is omitted, the
    service owns the created HTTP client — callers should use ``close()`` or a
    ``with`` block to release it.
    """

    cfg = settings or get_settings()
    hint = parse_embedding_provider(cfg.embedding_provider)
    owns_client = False
    http_client = client

    if hint is EmbeddingProviderHint.VOYAGE:
        owns_client = client is None
        http_client = client or create_http_client(timeout=cfg.embedding_timeout_seconds)

    provider = build_provider(
        cfg,
        client=http_client,
        local_embed_fn=local_embed_fn,
    )
    logger.info(
        "embedding service created provider=%s model=%s available=%s",
        provider.name.value,
        provider.model,
        provider.available,
        extra={
            "provider": provider.name.value,
            "model": provider.model,
            "available": provider.available,
        },
    )
    return EmbeddingService(
        provider,
        client=http_client,
        owns_client=owns_client,
    )
