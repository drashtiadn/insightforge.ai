"""Unit tests for search providers and SearchService (mocked HTTP)."""

from __future__ import annotations

import httpx
import pytest

from insightforge.agents.planner.schemas import ResearchTask
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.infrastructure.search import (
    ArxivSearchProvider,
    GitHubSearchProvider,
    RedditSearchProvider,
    SearchService,
    WebSearchProvider,
    WikipediaSearchProvider,
    YouTubeSearchProvider,
)
from insightforge.infrastructure.search.http import require_query
from insightforge.shared.enums import SearchProviderHint

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <summary>Transformer architecture paper.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <link href="https://arxiv.org/abs/1706.03762" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""


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


def test_web_tavily_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.tavily.com"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "LangGraph",
                        "url": "https://example.com/langgraph",
                        "content": "A graph library for agents.",
                        "score": 0.9,
                    }
                ]
            },
        )

    provider = WebSearchProvider(
        _client(httpx.MockTransport(handler)),
        tavily_api_key="tvly-test",
    )
    docs = provider.search("langgraph", limit=3)

    assert provider.available
    assert len(docs) == 1
    assert docs[0].title == "LangGraph"
    assert docs[0].provider is SearchProviderHint.WEB
    assert docs[0].metadata["backend"] == "tavily"


def test_web_falls_back_to_searxng_without_tavily() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/search" in str(request.url)
        assert request.url.params["q"] == "fastapi"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "FastAPI",
                        "url": "https://fastapi.tiangolo.com",
                        "content": "Modern Python web framework",
                    }
                ]
            },
        )

    provider = WebSearchProvider(
        _client(httpx.MockTransport(handler)),
        searxng_base_url="http://searx.local",
    )
    docs = provider.search("fastapi")

    assert len(docs) == 1
    assert docs[0].metadata["backend"] == "searxng"


def test_web_unavailable_without_config() -> None:
    provider = WebSearchProvider(_client(httpx.Response(200, json={})))
    assert not provider.available
    assert provider.search("anything") == []


def test_github_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.github.com" in str(request.url)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "full_name": "langchain-ai/langgraph",
                        "html_url": "https://github.com/langchain-ai/langgraph",
                        "description": "Build stateful agents",
                        "stargazers_count": 100,
                        "language": "Python",
                        "forks_count": 10,
                    }
                ]
            },
        )

    docs = GitHubSearchProvider(_client(httpx.MockTransport(handler))).search("langgraph")
    assert docs[0].title == "langchain-ai/langgraph"
    assert docs[0].provider is SearchProviderHint.GITHUB


def test_arxiv_search_parses_atom() -> None:
    docs = ArxivSearchProvider(_client(httpx.Response(200, text=ARXIV_ATOM))).search("transformers")
    assert len(docs) == 1
    assert docs[0].title == "Attention Is All You Need"
    assert "1706.03762" in docs[0].url
    assert docs[0].provider is SearchProviderHint.ARXIV


def test_wikipedia_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "title": "Photosynthesis",
                            "snippet": "Process used by <span>plants</span>",
                            "pageid": 1,
                        }
                    ]
                }
            },
        )

    docs = WikipediaSearchProvider(_client(httpx.MockTransport(handler))).search("photosynthesis")
    assert docs[0].title == "Photosynthesis"
    assert "wikipedia.org/wiki/Photosynthesis" in docs[0].url
    assert "<span>" not in docs[0].snippet


def test_youtube_requires_api_key() -> None:
    provider = YouTubeSearchProvider(_client(httpx.Response(200, json={})))
    assert not provider.available
    assert provider.search("agents") == []


def test_youtube_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "yt-key"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": {"videoId": "abc123"},
                        "snippet": {
                            "title": "Intro to Agents",
                            "description": "A tutorial",
                            "channelTitle": "AI Channel",
                            "publishedAt": "2024-01-01T00:00:00Z",
                        },
                    }
                ]
            },
        )

    docs = YouTubeSearchProvider(
        _client(httpx.MockTransport(handler)),
        api_key="yt-key",
    ).search("agents")
    assert docs[0].url == "https://www.youtube.com/watch?v=abc123"
    assert docs[0].provider is SearchProviderHint.YOUTUBE


def test_reddit_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["User-Agent"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Best RAG approaches?",
                                "permalink": "/r/MachineLearning/comments/1/rag/",
                                "url": "https://reddit.com/r/MachineLearning/comments/1/rag/",
                                "selftext": "Looking for tips",
                                "subreddit": "MachineLearning",
                                "score": 42,
                                "num_comments": 7,
                            }
                        }
                    ]
                }
            },
        )

    docs = RedditSearchProvider(_client(httpx.MockTransport(handler))).search("RAG")
    assert docs[0].provider is SearchProviderHint.REDDIT
    assert docs[0].url.startswith("https://www.reddit.com/r/")


def test_provider_http_error_becomes_external_service_error() -> None:
    provider = WikipediaSearchProvider(_client(httpx.Response(503, text="down")))
    with pytest.raises(ExternalServiceError, match="wikipedia"):
        provider.search("climate")


def test_search_service_aggregates_providers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""
        if "wikipedia" in host:
            return httpx.Response(
                200,
                json={
                    "query": {
                        "search": [
                            {
                                "title": "Climate",
                                "snippet": "Weather over time",
                                "pageid": 2,
                            }
                        ]
                    }
                },
            )
        if "reddit.com" in host:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "children": [
                            {
                                "data": {
                                    "title": "Climate discussion",
                                    "permalink": "/r/science/comments/1/c/",
                                    "url": "https://reddit.com",
                                    "selftext": "",
                                    "subreddit": "science",
                                    "score": 1,
                                    "num_comments": 0,
                                }
                            }
                        ]
                    }
                },
            )
        return httpx.Response(404, text="unexpected")

    client = _client(httpx.MockTransport(handler))
    service = SearchService(
        {
            SearchProviderHint.WIKIPEDIA: WikipediaSearchProvider(client),
            SearchProviderHint.REDDIT: RedditSearchProvider(client),
            SearchProviderHint.WEB: WebSearchProvider(client),
        }
    )

    docs = service.search(
        "climate",
        [SearchProviderHint.WIKIPEDIA, SearchProviderHint.REDDIT, SearchProviderHint.WEB],
    )
    assert len(docs) == 2
    assert {doc.provider for doc in docs} == {
        SearchProviderHint.WIKIPEDIA,
        SearchProviderHint.REDDIT,
    }


def test_search_service_soft_fails_provider_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "wikipedia" in (request.url.host or ""):
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "ok",
                                "permalink": "/r/test/comments/1/x/",
                                "url": "https://reddit.com",
                                "selftext": "body",
                                "subreddit": "test",
                                "score": 1,
                                "num_comments": 0,
                            }
                        }
                    ]
                }
            },
        )

    client = _client(httpx.MockTransport(handler))
    service = SearchService(
        {
            SearchProviderHint.WIKIPEDIA: WikipediaSearchProvider(client),
            SearchProviderHint.REDDIT: RedditSearchProvider(client),
        }
    )
    docs = service.search(
        "topic",
        [SearchProviderHint.WIKIPEDIA, SearchProviderHint.REDDIT],
    )
    assert len(docs) == 1
    assert docs[0].provider is SearchProviderHint.REDDIT


def test_search_task_uses_planner_task() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"query": {"search": [{"title": "RAG", "snippet": "Retrieval", "pageid": 9}]}},
        )

    client = _client(httpx.MockTransport(handler))
    service = SearchService({SearchProviderHint.WIKIPEDIA: WikipediaSearchProvider(client)})
    task = ResearchTask(
        id="t1",
        description="Collect background",
        search_query="retrieval augmented generation",
        providers=[SearchProviderHint.WIKIPEDIA],
        priority=1,
    )
    docs = service.search_task(task)
    assert docs[0].title == "RAG"


def test_search_rejects_empty_provider_list() -> None:
    service = SearchService({})
    with pytest.raises(ValidationFailedError, match="provider"):
        service.search("q", [])
