"""Unit tests for embedding providers and EmbeddingService."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import pytest

from insightforge.core.config import Settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.infrastructure.embeddings import (
    EmbeddingService,
    LocalEmbeddingProvider,
    VoyageEmbeddingProvider,
    create_embedding_service,
    parse_embedding_provider,
    require_texts,
)
from insightforge.shared.enums import EmbeddingInputType, EmbeddingProviderHint


def _client(handler: httpx.MockTransport | httpx.Response) -> httpx.Client:
    if isinstance(handler, httpx.Response):

        def _respond(_request: httpx.Request) -> httpx.Response:
            return handler

        transport: httpx.MockTransport = httpx.MockTransport(_respond)
    else:
        transport = handler
    return httpx.Client(transport=transport)


def test_require_texts_rejects_blank() -> None:
    with pytest.raises(ValidationFailedError):
        require_texts(["ok", "  "])


def test_parse_embedding_provider() -> None:
    assert parse_embedding_provider("voyage") is EmbeddingProviderHint.VOYAGE
    assert parse_embedding_provider("LOCAL") is EmbeddingProviderHint.LOCAL
    with pytest.raises(ValidationFailedError):
        parse_embedding_provider("openai")


def test_voyage_embed_batches_and_orders() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.voyageai.com"
        assert request.headers["Authorization"] == "Bearer voyage-test"
        payload = json.loads(request.content.decode())
        calls.append(payload)
        texts = payload["input"]
        assert isinstance(texts, list)
        data = [
            {"object": "embedding", "embedding": [float(i), 0.5], "index": i}
            for i, _ in enumerate(texts)
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": "voyage-3.5",
                "usage": {"total_tokens": len(texts) * 2},
            },
        )

    provider = VoyageEmbeddingProvider(
        _client(httpx.MockTransport(handler)),
        api_key="voyage-test",
        model="voyage-3.5",
        batch_size=2,
    )
    result = provider.embed(["a", "b", "c"], input_type=EmbeddingInputType.QUERY)

    assert provider.available
    assert result.provider is EmbeddingProviderHint.VOYAGE
    assert result.dimensions == 2
    assert result.usage_tokens == 6
    assert result.vectors == [[0.0, 0.5], [1.0, 0.5], [0.0, 0.5]]
    assert len(calls) == 2
    assert calls[0]["input_type"] == "query"
    assert calls[0]["input"] == ["a", "b"]
    assert calls[1]["input"] == ["c"]


def test_voyage_embed_passes_output_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["output_dimension"] == 256
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1] * 256, "index": 0}],
                "usage": {"total_tokens": 3},
            },
        )

    provider = VoyageEmbeddingProvider(
        _client(httpx.MockTransport(handler)),
        api_key="voyage-test",
        dimensions=256,
    )
    result = provider.embed(["hello"])
    assert result.dimensions == 256
    assert len(result.vectors[0]) == 256


def test_voyage_unavailable_without_key() -> None:
    provider = VoyageEmbeddingProvider(_client(httpx.Response(200)), api_key=None)
    assert not provider.available
    with pytest.raises(ExternalServiceError, match="VOYAGE_API_KEY"):
        provider.embed(["text"])


def test_voyage_http_error() -> None:
    provider = VoyageEmbeddingProvider(
        _client(httpx.Response(401, json={"detail": "unauthorized"})),
        api_key="bad",
    )
    with pytest.raises(ExternalServiceError, match="status 401"):
        provider.embed(["text"])


def test_local_embed_uses_injected_fn() -> None:
    def fake_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    provider = LocalEmbeddingProvider(embed_fn=fake_embed, batch_size=2)
    result = provider.embed(["ab", "cde", "f"])

    assert provider.available
    assert result.provider is EmbeddingProviderHint.LOCAL
    assert result.vectors == [[2.0, 1.0], [3.0, 1.0], [1.0, 1.0]]
    assert result.dimensions == 2


def test_local_empty_input() -> None:
    provider = LocalEmbeddingProvider(embed_fn=lambda texts: [])
    result = provider.embed([])
    assert result.vectors == []
    assert result.dimensions == 0


def test_embedding_service_query_and_documents() -> None:
    def fake_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]

    service = EmbeddingService(LocalEmbeddingProvider(embed_fn=fake_embed))
    docs = service.embed_documents(["doc one", "doc two"])
    query = service.embed_query("what is rag?")

    assert len(docs.vectors) == 2
    assert docs.input_type is EmbeddingInputType.DOCUMENT
    assert query == [1.0, 0.0]


def test_as_embed_fn_adapter() -> None:
    def fake_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.25, 0.75] for _ in texts]

    service = EmbeddingService(LocalEmbeddingProvider(embed_fn=fake_embed))
    vectors = service.as_embed_fn()(["alpha", "beta"])
    assert list(vectors) == [[0.25, 0.75], [0.25, 0.75]]


def test_create_embedding_service_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    settings = Settings(_env_file=None)

    def fake_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.0, 1.0] for _ in texts]

    with create_embedding_service(settings, local_embed_fn=fake_embed) as service:
        assert service.provider.name is EmbeddingProviderHint.LOCAL
        assert service.available
        result = service.embed(["hello"])
        assert result.vectors == [[0.0, 1.0]]


def test_create_embedding_service_voyage_owns_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-secret")
    settings = Settings(_env_file=None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.2, 0.8], "index": 0}], "usage": {"total_tokens": 1}},
        )

    client = _client(httpx.MockTransport(handler))
    service = create_embedding_service(settings, client=client)
    try:
        assert service.provider.name is EmbeddingProviderHint.VOYAGE
        assert service.available
        assert service.owns_client is False
        result = service.embed(["query text"], input_type=EmbeddingInputType.QUERY)
        assert result.vectors == [[0.2, 0.8]]
    finally:
        service.close()
        client.close()


def test_service_raises_when_unavailable() -> None:
    service = EmbeddingService(VoyageEmbeddingProvider(_client(httpx.Response(200))))
    with pytest.raises(ExternalServiceError, match="unavailable"):
        service.embed(["text"])
