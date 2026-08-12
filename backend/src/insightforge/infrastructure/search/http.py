"""Shared HTTP helpers for search providers."""

from __future__ import annotations

import httpx

from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(15.0)


def require_query(query: str) -> str:
    """Normalize and reject blank search queries."""

    cleaned = query.strip()
    if not cleaned:
        raise ValidationFailedError(
            "search query must not be empty",
            details={"field": "query"},
        )
    return cleaned


def create_http_client(*, timeout: float = 15.0) -> httpx.Client:
    """Build a short-lived sync client for provider calls."""

    return httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True)


def raise_for_status(response: httpx.Response, *, provider: str) -> None:
    """Map non-success HTTP responses to ``ExternalServiceError``."""

    if response.is_success:
        return
    logger.warning(
        "search provider HTTP error provider=%s status=%s",
        provider,
        response.status_code,
        extra={"provider": provider, "status_code": response.status_code},
    )
    raise ExternalServiceError(
        f"{provider} search failed with status {response.status_code}",
        details={
            "provider": provider,
            "status_code": response.status_code,
        },
    )
