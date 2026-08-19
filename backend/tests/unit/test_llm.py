"""Unit tests for the Gemini LLM client and JSON helpers."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from insightforge.core.config import Settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.infrastructure.llm import (
    GeminiLlmClient,
    LlmService,
    create_llm_service,
    extract_json_object,
    parse_llm_provider,
    parse_model,
)
from insightforge.shared.enums import LlmProviderHint


def _client(handler: httpx.MockTransport | httpx.Response) -> httpx.Client:
    if isinstance(handler, httpx.Response):

        def _respond(_request: httpx.Request) -> httpx.Response:
            return handler

        transport: httpx.MockTransport = httpx.MockTransport(_respond)
    else:
        transport = handler
    return httpx.Client(transport=transport)


def test_parse_llm_provider() -> None:
    assert parse_llm_provider("gemini") is LlmProviderHint.GEMINI
    assert parse_llm_provider("NONE") is LlmProviderHint.NONE
    with pytest.raises(ValidationFailedError):
        parse_llm_provider("openai")


def test_gemini_unavailable_without_key() -> None:
    client = GeminiLlmClient(None, api_key=None)
    assert not client.available
    with pytest.raises(ExternalServiceError, match="unavailable"):
        client.complete([{"role": "user", "content": "hi"}])


def test_create_llm_service_offline_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    service = create_llm_service(settings)
    assert not service.available
    service.close()


def test_gemini_generate_content_json_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "generativelanguage.googleapis.com"
        assert "gemini-2.5-flash" in str(request.url)
        assert request.headers["x-goog-api-key"] == "gem-test"
        payload = json.loads(request.content.decode())
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert payload["contents"][0]["role"] == "user"
        assert "systemInstruction" in payload
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}], "role": "model"}}]
            },
        )

    provider = GeminiLlmClient(
        _client(httpx.MockTransport(handler)),
        api_key="gem-test",
        model="gemini-2.5-flash",
    )
    text = provider.complete(
        [
            {"role": "system", "content": "json only"},
            {"role": "user", "content": "hello"},
        ],
        json_mode=True,
    )
    assert text == '{"ok": true}'


def test_llm_service_complete() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "pong"}]}}]},
        )

    service = LlmService(
        GeminiLlmClient(
            _client(httpx.MockTransport(handler)),
            api_key="gem-test",
        )
    )
    assert service.complete([{"role": "user", "content": "ping"}]) == "pong"


def test_extract_json_object_from_fence() -> None:
    payload = extract_json_object('```json\n{"a": 1}\n```')
    assert payload == {"a": 1}


def test_extract_json_object_rejects_array() -> None:
    with pytest.raises(ExternalServiceError):
        extract_json_object("[1, 2]")


class _Box(BaseModel):
    n: int


def test_parse_model() -> None:
    assert parse_model('{"n": 3}', _Box).n == 3
    with pytest.raises(ExternalServiceError):
        parse_model('{"n": "nope"}', _Box)
