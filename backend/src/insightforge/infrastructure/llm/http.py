"""Shared HTTP helpers for the Gemini LLM client."""

from __future__ import annotations

from typing import Any

import httpx

from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(60.0)


def create_http_client(*, timeout: float = 60.0) -> httpx.Client:
    """Build a sync client for Gemini generateContent calls."""

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
        "llm HTTP error provider=%s status=%s",
        provider,
        response.status_code,
        extra={"provider": provider, "status_code": response.status_code},
    )
    raise ExternalServiceError(
        f"{provider} llm failed with status {response.status_code}",
        details={
            "provider": provider,
            "status_code": response.status_code,
        },
    )


def candidate_text(payload: dict[str, Any]) -> str:
    """Extract the first candidate's concatenated text parts."""

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ExternalServiceError(
            "gemini returned no candidates",
            details={"provider": "gemini"},
        )
    first = candidates[0]
    if not isinstance(first, dict):
        raise ExternalServiceError(
            "gemini candidate is not an object",
            details={"provider": "gemini"},
        )
    content = first.get("content")
    if not isinstance(content, dict):
        raise ExternalServiceError(
            "gemini candidate is missing content",
            details={"provider": "gemini"},
        )
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ExternalServiceError(
            "gemini candidate has no text parts",
            details={"provider": "gemini"},
        )
    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    joined = "".join(texts).strip()
    if not joined:
        raise ExternalServiceError(
            "gemini candidate text is empty",
            details={"provider": "gemini"},
        )
    return joined
