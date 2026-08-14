"""Unit tests for reranker providers and RerankerService."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import pytest

from insightforge.agents.reranker import SimpleReranker
from insightforge.core.config import Settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.domain.models import RetrievalHit
from insightforge.infrastructure.rerankers import (
    BGEReranker,
    CrossEncoderReranker,
    JinaReranker,
    RerankerService,
    create_reranker_service,
    parse_reranker,
    require_query,
)
from insightforge.shared.enums import RerankerHint, RetrievalMode


def _hit(record_id: str, text: str, score: float = 0.5) -> RetrievalHit:
    return RetrievalHit(
        id=record_id,
        score=score,
        text=text,
        mode=RetrievalMode.HYBRID,
        metadata={"source": "test"},
    )


def _prefer_rag(query: str, documents: Sequence[str]) -> list[float]:
    needle = "rag" if "rag" in query.lower() else query.lower()
    return [1.0 if needle in doc.lower() else 0.1 for doc in documents]


def _client(handler: httpx.MockTransport | httpx.Response) -> httpx.Client:
    if isinstance(handler, httpx.Response):

        def _respond(_request: httpx.Request) -> httpx.Response:
            return handler

        transport: httpx.MockTransport = httpx.MockTransport(_respond)
    else:
        transport = handler
    return httpx.Client(transport=transport)


def test_require_query_rejects_blank() -> None:
    with pytest.raises(ValidationFailedError):
        require_query("  ")


def test_parse_reranker() -> None:
    assert parse_reranker("cross-encoder") is RerankerHint.CROSS_ENCODER
    assert parse_reranker("BGE") is RerankerHint.BGE
    assert parse_reranker("jina") is RerankerHint.JINA
    with pytest.raises(ValidationFailedError):
        parse_reranker("cohere")


def test_cross_encoder_injected_scorer_reorders() -> None:
    provider = CrossEncoderReranker(score_fn=_prefer_rag)
    hits = [
        _hit("noise", "Cats sit on mats.", score=0.9),
        _hit("keep", "RAG retrieves relevant context.", score=0.2),
    ]
    ranked = provider.rerank("what is RAG?", hits)

    assert provider.available
    assert provider.name is RerankerHint.CROSS_ENCODER
    assert [hit.id for hit in ranked] == ["keep", "noise"]
    assert ranked[0].rerank_rank == 1
    assert ranked[0].retrieval_score == 0.2
    assert ranked[0].score == pytest.approx(1.0)


def test_bge_top_n_and_empty() -> None:
    provider = BGEReranker(score_fn=_prefer_rag)
    hits = [
        _hit("a", "unrelated weather"),
        _hit("b", "RAG embeddings"),
        _hit("c", "also RAG pipelines"),
    ]
    ranked = provider.rerank("RAG", hits, top_n=2)
    assert provider.name is RerankerHint.BGE
    assert len(ranked) == 2
    assert ranked[0].id in {"b", "c"}
    assert provider.rerank("RAG", []) == []


def test_jina_rerank_orders_by_api_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.jina.ai"
        assert request.headers["Authorization"] == "Bearer jina-test"
        payload = json.loads(request.content.decode())
        assert payload["model"] == "jina-reranker-v2-base-multilingual"
        assert payload["query"] == "RAG"
        assert payload["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.10},
                ]
            },
        )

    provider = JinaReranker(
        _client(httpx.MockTransport(handler)),
        api_key="jina-test",
    )
    ranked = provider.rerank(
        "RAG",
        [_hit("first", "noise"), _hit("second", "RAG context")],
        top_n=2,
    )
    assert [hit.id for hit in ranked] == ["second", "first"]
    assert ranked[0].score == pytest.approx(0.95)


def test_jina_unavailable_without_key() -> None:
    provider = JinaReranker(_client(httpx.Response(200)), api_key=None)
    assert not provider.available
    with pytest.raises(ExternalServiceError, match="JINA_API_KEY"):
        provider.rerank("q", [_hit("a", "text")])


def test_jina_http_error() -> None:
    provider = JinaReranker(
        _client(httpx.Response(401, json={"detail": "unauthorized"})),
        api_key="bad",
    )
    with pytest.raises(ExternalServiceError, match="status 401"):
        provider.rerank("q", [_hit("a", "text")])


def test_reranker_service_and_agent() -> None:
    service = RerankerService(CrossEncoderReranker(score_fn=_prefer_rag), default_top_n=1)
    hits = [_hit("a", "cats"), _hit("b", "RAG")]
    ranked = SimpleReranker(service).run("RAG", hits)
    assert len(ranked) == 1
    assert ranked[0].id == "b"


def test_create_reranker_service_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_PROVIDER", "cross-encoder")
    settings = Settings(_env_file=None)
    with create_reranker_service(settings, score_fn=_prefer_rag) as service:
        assert service.provider.name is RerankerHint.CROSS_ENCODER
        assert service.available
        ranked = service.rerank("RAG", [_hit("x", "RAG"), _hit("y", "zzz")])
        assert ranked[0].id == "x"


def test_create_reranker_service_jina(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_PROVIDER", "jina")
    monkeypatch.setenv("JINA_API_KEY", "jina-secret")
    settings = Settings(_env_file=None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.7}]},
        )

    client = _client(httpx.MockTransport(handler))
    service = create_reranker_service(settings, client=client)
    try:
        assert service.provider.name is RerankerHint.JINA
        ranked = service.rerank("q", [_hit("only", "doc")])
        assert ranked[0].id == "only"
        assert ranked[0].score == pytest.approx(0.7)
    finally:
        service.close()
        client.close()


def test_service_raises_when_unavailable() -> None:
    service = RerankerService(JinaReranker(_client(httpx.Response(200))))
    with pytest.raises(ExternalServiceError, match="unavailable"):
        service.rerank("q", [_hit("a", "text")])
