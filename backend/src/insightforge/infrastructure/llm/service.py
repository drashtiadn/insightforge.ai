"""LLM service — select Gemini, complete prompts, close the HTTP client."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

import httpx

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.infrastructure.llm.base import LlmClient, LlmMessage
from insightforge.infrastructure.llm.gemini import GeminiLlmClient
from insightforge.infrastructure.llm.http import close_http_client, create_http_client
from insightforge.shared.enums import LlmProviderHint

logger = get_logger(__name__)


def parse_llm_provider(value: str) -> LlmProviderHint:
    """Parse settings/env value into ``LlmProviderHint``."""

    cleaned = value.strip().lower()
    try:
        return LlmProviderHint(cleaned)
    except ValueError as exc:
        allowed = ", ".join(hint.value for hint in LlmProviderHint)
        raise ValidationFailedError(
            f"unknown llm provider {value!r}; expected one of: {allowed}",
            details={"field": "llm_provider", "value": value},
        ) from exc


def build_client(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
) -> LlmClient:
    """Construct the configured LLM client."""

    hint = parse_llm_provider(settings.llm_provider)
    api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    if hint is LlmProviderHint.NONE:
        return GeminiLlmClient(
            None,
            api_key=None,
            model=settings.llm_gemini_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    return GeminiLlmClient(
        client,
        api_key=api_key,
        model=settings.llm_gemini_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


class LlmService:
    """Facade over a single LLM client.

    When constructed via ``create_llm_service`` for Gemini with a key and
    without an injected client, this service owns the ``httpx.Client``.
    """

    def __init__(
        self,
        provider: LlmClient,
        *,
        client: httpx.Client | None = None,
        owns_client: bool = False,
    ) -> None:
        self._provider = provider
        self._client = client
        self._owns_client = owns_client and client is not None

    @property
    def provider(self) -> LlmClient:
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

    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        json_mode: bool = False,
    ) -> str:
        """Return assistant text, raising if the provider is unavailable."""

        if not self._provider.available:
            raise ExternalServiceError(
                f"llm unavailable: {self._provider.name.value}",
                details={"provider": self._provider.name.value},
            )
        logger.info(
            "llm complete provider=%s model=%s json_mode=%s",
            self._provider.name.value,
            self._provider.model,
            json_mode,
        )
        return self._provider.complete(messages, json_mode=json_mode)


def create_llm_service(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
) -> LlmService:
    """Factory used by the graph and tests.

    Owns an HTTP client only when Gemini is selected and an API key is set.
    """

    cfg = settings or get_settings()
    hint = parse_llm_provider(cfg.llm_provider)
    has_key = cfg.gemini_api_key is not None
    owns_client = False
    http_client = client

    if hint is LlmProviderHint.GEMINI and has_key and client is None:
        owns_client = True
        http_client = create_http_client(timeout=cfg.llm_timeout_seconds)

    provider = build_client(cfg, client=http_client)
    logger.info(
        "llm service created provider=%s model=%s available=%s",
        provider.name.value,
        provider.model,
        provider.available,
        extra={
            "provider": provider.name.value,
            "model": provider.model,
            "available": provider.available,
        },
    )
    return LlmService(provider, client=http_client, owns_client=owns_client)
