"""Web search via Tavily (preferred) or SearXNG (self-hosted fallback)."""

from __future__ import annotations

from typing import Any

import httpx

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.http import raise_for_status, require_query
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)


class WebSearchProvider(SearchProvider):
    """``WEB`` hint — Tavily when keyed, otherwise SearXNG."""

    name = SearchProviderHint.WEB

    def __init__(
        self,
        client: httpx.Client,
        *,
        tavily_api_key: str | None = None,
        searxng_base_url: str | None = None,
    ) -> None:
        self._client = client
        self._tavily_api_key = (tavily_api_key or "").strip() or None
        base = (searxng_base_url or "").strip().rstrip("/")
        self._searxng_base_url = base or None

    @property
    def available(self) -> bool:
        return bool(self._tavily_api_key or self._searxng_base_url)

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        cleaned = require_query(query)
        if self._tavily_api_key:
            return self._search_tavily(cleaned, limit=limit)
        if self._searxng_base_url:
            return self._search_searxng(cleaned, limit=limit)
        logger.warning("web search skipped: no Tavily API key or SearXNG URL configured")
        return []

    def _search_tavily(self, query: str, *, limit: int) -> list[Document]:
        logger.info("web search via tavily query_len=%d limit=%d", len(query), limit)
        response = self._client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self._tavily_api_key,
                "query": query,
                "max_results": limit,
                "include_answer": False,
            },
        )
        raise_for_status(response, provider="tavily")
        payload = response.json()
        results = payload.get("results") or []
        documents: list[Document] = []
        for item in results[:limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            content = str(item.get("content") or "").strip()
            documents.append(
                Document(
                    title=title,
                    url=url,
                    snippet=content[:400],
                    content=content,
                    provider=self.name,
                    metadata={"backend": "tavily", "score": item.get("score")},
                )
            )
        logger.info("tavily returned count=%d", len(documents))
        return documents

    def _search_searxng(self, query: str, *, limit: int) -> list[Document]:
        assert self._searxng_base_url is not None
        logger.info("web search via searxng query_len=%d limit=%d", len(query), limit)
        response = self._client.get(
            f"{self._searxng_base_url}/search",
            params={"q": query, "format": "json", "categories": "general"},
        )
        raise_for_status(response, provider="searxng")
        payload: dict[str, Any] = response.json()
        results = payload.get("results") or []
        documents: list[Document] = []
        for item in results[:limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            content = str(item.get("content") or item.get("snippet") or "").strip()
            documents.append(
                Document(
                    title=title,
                    url=url,
                    snippet=content[:400],
                    content=content,
                    provider=self.name,
                    metadata={"backend": "searxng", "engine": item.get("engine")},
                )
            )
        logger.info("searxng returned count=%d", len(documents))
        return documents
