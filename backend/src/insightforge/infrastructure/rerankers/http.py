"""Shared helpers for reranker providers."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit

logger = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(30.0)


def require_query(query: str) -> str:
    """Normalize and reject blank rerank queries."""

    cleaned = query.strip()
    if not cleaned:
        raise ValidationFailedError(
            "rerank query must not be empty",
            details={"field": "query"},
        )
    return cleaned


def create_http_client(*, timeout: float = 30.0) -> httpx.Client:
    """Build a sync client for remote rerankers."""

    return httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True)


def close_http_client(client: httpx.Client | None) -> None:
    """Close ``client`` if it is still open. Safe to call more than once."""

    if client is None:
        return
    if client.is_closed:
        return
    client.close()


def raise_for_status(response: httpx.Response, *, provider: str) -> None:
    """Map non-success HTTP responses to ``ExternalServiceError``."""

    if response.is_success:
        return
    logger.warning(
        "reranker HTTP error provider=%s status=%s",
        provider,
        response.status_code,
        extra={"provider": provider, "status_code": response.status_code},
    )
    raise ExternalServiceError(
        f"{provider} rerank failed with status {response.status_code}",
        details={
            "provider": provider,
            "status_code": response.status_code,
        },
    )


def apply_scores(
    hits: Sequence[RetrievalHit],
    scores: Sequence[float],
    *,
    top_n: int | None = None,
) -> list[RetrievalHit]:
    """Attach rerank scores, sort descending, and optionally trim to ``top_n``."""

    if len(scores) != len(hits):
        raise ExternalServiceError(
            "reranker returned unexpected score count",
            details={"expected": len(hits), "actual": len(scores)},
        )

    ranked = [
        hit.model_copy(
            update={
                "score": float(score),
                "retrieval_score": (
                    hit.score if hit.retrieval_score is None else hit.retrieval_score
                ),
            }
        )
        for hit, score in zip(hits, scores, strict=True)
    ]
    ranked.sort(key=lambda hit: hit.score, reverse=True)
    if top_n is not None:
        if top_n < 1:
            return []
        ranked = ranked[:top_n]
    return [
        hit.model_copy(update={"rerank_rank": index}) for index, hit in enumerate(ranked, start=1)
    ]
