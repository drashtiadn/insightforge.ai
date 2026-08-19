"""Tests for Phase 3.3 search features: parallel, dedupe, scoring, rate limit."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from insightforge.domain.models import Document, ResearchTask
from insightforge.graph.nodes import search_node
from insightforge.graph.state import initial_state
from insightforge.infrastructure.search import (
    RateLimiter,
    SearchProvider,
    SearchService,
    create_search_service,
    dedupe_documents,
    normalize_url,
    provider_metadata_score,
    score_document,
    score_documents,
)
from insightforge.shared.enums import SearchProviderHint


class _FakeProvider(SearchProvider):
    def __init__(
        self,
        name: SearchProviderHint,
        docs: list[Document],
        *,
        delay: float = 0.0,
        fail: bool = False,
        calls: list[str] | None = None,
    ) -> None:
        self.name = name
        self._docs = docs
        self._delay = delay
        self._fail = fail
        self._calls = calls if calls is not None else []

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        self._calls.append(self.name.value)
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.name.value} boom")
        return self._docs[:limit]


def _doc(
    title: str,
    url: str,
    provider: SearchProviderHint,
    *,
    snippet: str = "",
    metadata: dict[str, Any] | None = None,
    score: float | None = None,
) -> Document:
    return Document(
        title=title,
        url=url,
        snippet=snippet,
        content=snippet,
        provider=provider,
        score=score,
        metadata=metadata or {},
    )


def test_normalize_url_strips_tracking_and_www() -> None:
    left = normalize_url("https://WWW.Example.com/path/?utm_source=x&q=1")
    right = normalize_url("https://example.com/path?q=1")
    assert left == right


def test_dedupe_keeps_higher_score() -> None:
    docs = [
        _doc("A", "https://example.com/a", SearchProviderHint.WEB, score=0.2),
        _doc(
            "A better",
            "https://example.com/a?utm_campaign=1",
            SearchProviderHint.WIKIPEDIA,
            score=0.9,
            snippet="richer",
        ),
    ]
    result = dedupe_documents(docs)
    assert len(result) == 1
    assert result[0].title == "A better"
    assert result[0].score == 0.9


def test_score_documents_orders_by_relevance() -> None:
    docs = [
        _doc(
            "Unrelated cats",
            "https://example.com/cats",
            SearchProviderHint.REDDIT,
            snippet="meow",
        ),
        _doc(
            "LangGraph agents",
            "https://example.com/langgraph",
            SearchProviderHint.WEB,
            snippet="LangGraph multi agent systems",
            metadata={"score": 0.95},
        ),
    ]
    scored = score_documents(docs, "LangGraph agents")
    assert scored[0].title == "LangGraph agents"
    assert scored[0].score is not None
    assert scored[0].score > (scored[1].score or 0.0)


def test_score_document_ignores_provider_identity() -> None:
    arxiv = _doc("Paper", "https://arxiv.org/abs/1", SearchProviderHint.ARXIV, snippet="x")
    reddit = _doc("Post", "https://reddit.com/r/x", SearchProviderHint.REDDIT, snippet="x")
    assert score_document(arxiv, "topic") == score_document(reddit, "topic")


def test_score_document_ignores_provider_metadata_score() -> None:
    plain = _doc(
        "LangGraph guide",
        "https://example.com/a",
        SearchProviderHint.WEB,
        snippet="LangGraph agents",
    )
    boosted = _doc(
        "LangGraph guide",
        "https://example.com/b",
        SearchProviderHint.WEB,
        snippet="LangGraph agents",
        metadata={"score": 0.99, "stars": 9000},
    )
    assert score_document(plain, "LangGraph agents") == score_document(boosted, "LangGraph agents")
    assert provider_metadata_score(boosted) is not None
    assert provider_metadata_score(plain) is None


def test_rate_limiter_enforces_interval() -> None:
    limiter = RateLimiter(calls_per_second=10.0)  # 100ms gap
    start = time.monotonic()
    limiter.acquire("web")
    waited = limiter.acquire("web")
    elapsed = time.monotonic() - start
    assert waited >= 0.05
    assert elapsed >= 0.05


def test_rate_limiter_is_thread_safe() -> None:
    limiter = RateLimiter(calls_per_second=50.0)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(5):
                limiter.acquire("github")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_search_service_runs_providers_in_parallel() -> None:
    lock = threading.Lock()
    active = {"count": 0, "max": 0}

    class _ConcurrentProvider(_FakeProvider):
        def search(self, query: str, *, limit: int = 5) -> list[Document]:
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            try:
                return super().search(query, limit=limit)
            finally:
                with lock:
                    active["count"] -= 1

    providers = {
        SearchProviderHint.WEB: _ConcurrentProvider(
            SearchProviderHint.WEB,
            [_doc("W", "https://example.com/w", SearchProviderHint.WEB, snippet="langgraph")],
            delay=0.1,
        ),
        SearchProviderHint.WIKIPEDIA: _ConcurrentProvider(
            SearchProviderHint.WIKIPEDIA,
            [
                _doc(
                    "Wiki",
                    "https://example.com/wiki",
                    SearchProviderHint.WIKIPEDIA,
                    snippet="langgraph",
                )
            ],
            delay=0.1,
        ),
    }
    service = SearchService(providers, max_workers=2, dedupe=True, scoring=True)
    docs = service.search(
        "langgraph",
        [SearchProviderHint.WEB, SearchProviderHint.WIKIPEDIA],
    )

    assert len(docs) == 2
    assert active["max"] >= 2


def test_search_service_soft_fails_and_dedupes() -> None:
    providers = {
        SearchProviderHint.WEB: _FakeProvider(
            SearchProviderHint.WEB,
            [_doc("Same", "https://example.com/x", SearchProviderHint.WEB, snippet="q")],
            fail=True,
        ),
        SearchProviderHint.GITHUB: _FakeProvider(
            SearchProviderHint.GITHUB,
            [
                _doc("Same", "https://example.com/x", SearchProviderHint.GITHUB, snippet="q"),
                _doc("Other", "https://example.com/y", SearchProviderHint.GITHUB, snippet="q"),
            ],
        ),
    }
    service = SearchService(providers, dedupe=True, scoring=True, max_documents=10)
    docs = service.search("q", [SearchProviderHint.WEB, SearchProviderHint.GITHUB])
    assert len(docs) == 2
    urls = {doc.url for doc in docs}
    assert urls == {"https://example.com/x", "https://example.com/y"}


def test_search_service_trims_to_max_documents() -> None:
    docs = [
        _doc(f"D{i}", f"https://example.com/{i}", SearchProviderHint.WEB, snippet="topic")
        for i in range(10)
    ]
    providers = {SearchProviderHint.WEB: _FakeProvider(SearchProviderHint.WEB, docs)}
    service = SearchService(providers, max_documents=3, scoring=True)
    assert len(service.search("topic", [SearchProviderHint.WEB], limit=10)) == 3


def test_search_tasks_merges_across_planner_tasks() -> None:
    providers = {
        SearchProviderHint.WIKIPEDIA: _FakeProvider(
            SearchProviderHint.WIKIPEDIA,
            [_doc("A", "https://example.com/a", SearchProviderHint.WIKIPEDIA, snippet="rag")],
        ),
        SearchProviderHint.WEB: _FakeProvider(
            SearchProviderHint.WEB,
            [_doc("B", "https://example.com/b", SearchProviderHint.WEB, snippet="rag")],
        ),
    }
    service = SearchService(providers, scoring=True)
    tasks = [
        ResearchTask(
            id="t1",
            description="wiki",
            search_query="rag",
            providers=[SearchProviderHint.WIKIPEDIA],
            priority=1,
        ),
        ResearchTask(
            id="t2",
            description="web",
            search_query="rag",
            providers=[SearchProviderHint.WEB],
            priority=2,
        ),
    ]
    docs = service.search_tasks(tasks)
    assert {doc.title for doc in docs} == {"A", "B"}


def test_search_tasks_scores_each_task_against_its_own_query() -> None:
    providers = {
        SearchProviderHint.WEB: _FakeProvider(
            SearchProviderHint.WEB,
            [
                _doc(
                    "Alpha hit",
                    "https://example.com/alpha",
                    SearchProviderHint.WEB,
                    snippet="alpha topic",
                )
            ],
        ),
        SearchProviderHint.WIKIPEDIA: _FakeProvider(
            SearchProviderHint.WIKIPEDIA,
            [
                _doc(
                    "Beta hit",
                    "https://example.com/beta",
                    SearchProviderHint.WIKIPEDIA,
                    snippet="beta topic",
                )
            ],
        ),
    }
    service = SearchService(providers, scoring=True, dedupe=False)
    tasks = [
        ResearchTask(
            id="t1",
            description="alpha",
            search_query="alpha",
            providers=[SearchProviderHint.WEB],
            priority=1,
        ),
        ResearchTask(
            id="t2",
            description="beta",
            search_query="beta",
            providers=[SearchProviderHint.WIKIPEDIA],
            priority=2,
        ),
    ]
    docs = {doc.title: doc for doc in service.search_tasks(tasks)}
    assert docs["Alpha hit"].score is not None
    assert docs["Beta hit"].score is not None
    assert docs["Alpha hit"].score > 0.5
    assert docs["Beta hit"].score > 0.5


def test_search_node_uses_search_service() -> None:
    provider = _FakeProvider(
        SearchProviderHint.WIKIPEDIA,
        [
            _doc(
                "Photosynthesis",
                "https://en.wikipedia.org/wiki/Photosynthesis",
                SearchProviderHint.WIKIPEDIA,
                snippet="plants",
            )
        ],
    )
    service = SearchService({SearchProviderHint.WIKIPEDIA: provider}, scoring=True)
    state = initial_state("what is photosynthesis")
    state["step"] = 1
    state["phase"] = "research"
    state["tasks"] = [
        ResearchTask(
            id="t1",
            description="Collect encyclopedic background",
            search_query="photosynthesis",
            providers=[SearchProviderHint.WIKIPEDIA],
            priority=1,
        ).model_dump(mode="json")
    ]

    result = search_node(state, search_service=service)
    assert result["phase"] == "search"
    assert len(result["documents"]) == 1
    assert result["documents"][0]["title"] == "Photosynthesis"
    assert result["sources"][0]["url"].endswith("Photosynthesis")
    assert result["documents"][0]["score"] is not None


def test_search_node_stub_path_without_search_service() -> None:
    state = initial_state("AI")
    state["step"] = 1
    state["phase"] = "research"
    result = search_node(state)
    assert result["sources"][0]["title"].startswith("Source 1")
    assert result["documents"][0]["title"].startswith("Source 1")
    assert "overview of AI" in result["documents"][0]["snippet"]


def test_create_search_service_owns_and_closes_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock(spec=httpx.Client)
    fake_client.is_closed = False
    monkeypatch.setattr(
        "insightforge.infrastructure.search.service.create_http_client",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setattr(
        "insightforge.infrastructure.search.service.build_providers",
        lambda _settings, _client: {},
    )

    service = create_search_service()
    assert service.owns_client is True
    assert service.client is fake_client
    service.close()
    fake_client.close.assert_called_once()
    assert service.client is None
    assert service.owns_client is False

    # Idempotent
    service.close()
    fake_client.close.assert_called_once()


def test_create_search_service_does_not_close_injected_client() -> None:
    injected = MagicMock(spec=httpx.Client)
    injected.is_closed = False
    service = SearchService({}, client=injected, owns_client=False)
    with service:
        pass
    injected.close.assert_not_called()
