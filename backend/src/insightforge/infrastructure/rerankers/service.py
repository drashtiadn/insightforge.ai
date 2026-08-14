"""Reranker service — select provider and reorder retrieval hits."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

import httpx

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit
from insightforge.infrastructure.rerankers.base import RerankerProvider
from insightforge.infrastructure.rerankers.http import close_http_client, create_http_client
from insightforge.infrastructure.rerankers.providers import (
    BGEReranker,
    CrossEncoderReranker,
    JinaReranker,
    RerankFn,
)
from insightforge.shared.enums import RerankerHint

logger = get_logger(__name__)


def parse_reranker(value: str) -> RerankerHint:
    """Parse settings/env value into ``RerankerHint``."""

    cleaned = value.strip().lower()
    try:
        return RerankerHint(cleaned)
    except ValueError as exc:
        allowed = ", ".join(hint.value for hint in RerankerHint)
        raise ValidationFailedError(
            f"unknown reranker {value!r}; expected one of: {allowed}",
            details={"field": "reranker_provider", "value": value},
        ) from exc


def build_provider(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    score_fn: RerankFn | None = None,
) -> RerankerProvider:
    """Construct the configured reranker provider."""

    hint = parse_reranker(settings.reranker_provider)
    if hint is RerankerHint.JINA:
        if client is None:
            raise ValidationFailedError(
                "jina rerank requires an httpx client",
                details={"field": "client"},
            )
        api_key = settings.jina_api_key.get_secret_value() if settings.jina_api_key else None
        return JinaReranker(
            client,
            api_key=api_key,
            model=settings.reranker_jina_model,
        )
    if hint is RerankerHint.BGE:
        return BGEReranker(model=settings.reranker_bge_model, score_fn=score_fn)
    return CrossEncoderReranker(
        model=settings.reranker_cross_encoder_model,
        score_fn=score_fn,
    )


class RerankerService:
    """Facade over a single reranker provider.

    When constructed via ``create_reranker_service`` for Jina without an
    injected client, this service owns the shared ``httpx.Client``.
    """

    def __init__(
        self,
        provider: RerankerProvider,
        *,
        default_top_n: int | None = None,
        client: httpx.Client | None = None,
        owns_client: bool = False,
    ) -> None:
        self._provider = provider
        self._default_top_n = default_top_n
        self._client = client
        self._owns_client = owns_client and client is not None

    @property
    def provider(self) -> RerankerProvider:
        return self._provider

    @property
    def available(self) -> bool:
        return self._provider.available

    @property
    def model(self) -> str:
        return self._provider.model

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

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        """Reorder retrieval hits by relevance to ``query``."""

        if not self._provider.available:
            raise ExternalServiceError(
                f"reranker unavailable: {self._provider.name.value}",
                details={"provider": self._provider.name.value},
            )

        limit = top_n if top_n is not None else self._default_top_n
        logger.info(
            "rerank started provider=%s model=%s candidates=%d top_n=%s",
            self._provider.name.value,
            self._provider.model,
            len(hits),
            limit,
            extra={
                "provider": self._provider.name.value,
                "candidates": len(hits),
            },
        )
        ranked = self._provider.rerank(query, hits, top_n=limit)
        logger.info(
            "rerank finished provider=%s count=%d",
            self._provider.name.value,
            len(ranked),
            extra={"provider": self._provider.name.value, "count": len(ranked)},
        )
        return ranked


def create_reranker_service(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
    score_fn: RerankFn | None = None,
) -> RerankerService:
    """Factory used by application code and tests.

    When the configured provider is Jina and ``client`` is omitted, the service
    owns the created HTTP client — use ``close()`` or a ``with`` block.
    """

    cfg = settings or get_settings()
    hint = parse_reranker(cfg.reranker_provider)
    owns_client = False
    http_client = client

    if hint is RerankerHint.JINA:
        owns_client = client is None
        http_client = client or create_http_client(timeout=cfg.reranker_timeout_seconds)

    provider = build_provider(cfg, client=http_client, score_fn=score_fn)
    logger.info(
        "reranker service created provider=%s model=%s available=%s",
        provider.name.value,
        provider.model,
        provider.available,
        extra={
            "provider": provider.name.value,
            "model": provider.model,
            "available": provider.available,
        },
    )
    return RerankerService(
        provider,
        default_top_n=cfg.reranker_top_n,
        client=http_client,
        owns_client=owns_client,
    )
