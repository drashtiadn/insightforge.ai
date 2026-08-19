"""Google Gemini generateContent client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import httpx

from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.infrastructure.llm.base import LlmClient, LlmMessage
from insightforge.infrastructure.llm.http import candidate_text, raise_for_status
from insightforge.shared.enums import LlmProviderHint

logger = get_logger(__name__)

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class GeminiLlmClient(LlmClient):
    """Remote chat completions via the Gemini REST API."""

    name = LlmProviderHint.GEMINI

    def __init__(
        self,
        client: httpx.Client | None,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self._client = client
        self._api_key = (api_key or "").strip() or None
        self._model = (model or "").strip() or "gemini-2.5-flash"
        self._temperature = temperature
        self._max_tokens = max(1, max_tokens)

    @property
    def available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        json_mode: bool = False,
    ) -> str:
        if not self.available or self._client is None:
            raise ExternalServiceError(
                "gemini llm is unavailable",
                details={"provider": self.name.value},
            )
        if not messages:
            raise ValidationFailedError(
                "llm messages must not be empty",
                details={"field": "messages"},
            )

        url = GEMINI_GENERATE_URL.format(model=quote(self._model, safe=".-"))
        payload = self._build_payload(messages, json_mode=json_mode)
        logger.info(
            "gemini complete model=%s json_mode=%s messages=%d",
            self._model,
            json_mode,
            len(messages),
        )
        try:
            response = self._client.post(
                url,
                headers={"x-goog-api-key": self._api_key or ""},
                json=payload,
            )
        except httpx.HTTPError as exc:
            logger.warning("gemini request failed error=%s", exc)
            raise ExternalServiceError(
                f"gemini llm request failed: {exc}",
                details={"provider": self.name.value},
            ) from exc

        raise_for_status(response, provider=self.name.value)
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                "gemini returned non-JSON",
                details={"provider": self.name.value},
            ) from exc
        if not isinstance(body, dict):
            raise ExternalServiceError(
                "gemini response is not an object",
                details={"provider": self.name.value},
            )
        return candidate_text(body)

    def _build_payload(
        self,
        messages: Sequence[LlmMessage],
        *,
        json_mode: bool,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"].strip().lower()
            content = message["content"]
            if role == "system":
                system_parts.append(content)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

        if not contents:
            raise ValidationFailedError(
                "llm messages must include a user turn",
                details={"field": "messages"},
            )

        generation: dict[str, Any] = {
            "temperature": self._temperature,
            "maxOutputTokens": self._max_tokens,
        }
        if json_mode:
            generation["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation,
        }
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}],
            }
        return payload
