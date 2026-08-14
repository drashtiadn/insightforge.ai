"""Shared helpers for embedding providers."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(30.0)


def require_texts(texts: Sequence[str]) -> list[str]:
    """Normalize texts and reject blank entries.

    An empty sequence is allowed (callers return an empty result). Individual
    items must be non-empty after stripping.
    """

    cleaned: list[str] = []
    for index, text in enumerate(texts):
        value = text.strip() if isinstance(text, str) else ""
        if not value:
            raise ValidationFailedError(
                "embedding text must not be empty",
                details={"field": "texts", "index": index},
            )
        cleaned.append(value)
    return cleaned


def create_http_client(*, timeout: float = 30.0) -> httpx.Client:
    """Build a sync client shared by remote embedding providers."""

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
        "embedding provider HTTP error provider=%s status=%s",
        provider,
        response.status_code,
        extra={"provider": provider, "status_code": response.status_code},
    )
    raise ExternalServiceError(
        f"{provider} embeddings failed with status {response.status_code}",
        details={
            "provider": provider,
            "status_code": response.status_code,
        },
    )


def batched(items: Sequence[str], size: int) -> list[list[str]]:
    """Split ``items`` into contiguous batches of at most ``size``."""

    if size < 1:
        raise ValidationFailedError(
            "embedding batch size must be >= 1",
            details={"field": "batch_size", "value": size},
        )
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
