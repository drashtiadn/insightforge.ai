"""Tests for Phase 3.3 search features: parallel, dedupe, scoring, rate limit."""

from __future__ import annotations

import threading
import time
from typing import Any

from insightforge.agents.planner.schemas import ResearchTask
from insightforge.domain.models import Document
from insightforge.graph.nodes import retrieve_node
from insightforge.graph.state import initial_state
from insightforge.infrastructure.search import (
    RateLimiter,
    SearchProvider,
    SearchService,
    dedupe_documents,
    normalize_url,
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


def test_score_document_uses_provider_weight() -> None:
    arxiv = _doc("Paper", "https://arxiv.org/abs/1", SearchProviderHint.ARXIV, snippet="x")
    reddit = _doc("Post", "https://reddit.com/r/x", SearchProviderHint.REDDIT, snippet="x")
    assert score_document(arxiv, "topic") > score_document(reddit, "topic")


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


def test_retrieve_node_uses_search_service() -> None:
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

    result = retrieve_node(state, search_service=service)
    assert result["phase"] == "retrieve"
    assert len(result["documents"]) == 1
    assert result["documents"][0]["title"] == "Photosynthesis"
    assert result["sources"][0]["url"].endswith("Photosynthesis")
    assert result["documents"][0]["score"] is not None


def test_retrieve_node_stub_path_without_search_service() -> None:
    state = initial_state("AI")
    state["step"] = 1
    state["phase"] = "research"
    result = retrieve_node(state)
    assert result["sources"][0]["title"].startswith("Source 1")
    assert "documents" not in result
