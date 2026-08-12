"""Reddit search via the public JSON API."""

from __future__ import annotations

import httpx

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.http import raise_for_status, require_query
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)


class RedditSearchProvider(SearchProvider):
    """Search Reddit posts. No API key required; User-Agent is mandatory."""

    name = SearchProviderHint.REDDIT

    def __init__(
        self,
        client: httpx.Client,
        *,
        user_agent: str = "insightforge/0.1.0 (research platform)",
    ) -> None:
        self._client = client
        self._user_agent = user_agent

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        cleaned = require_query(query)
        logger.info("reddit search query_len=%d limit=%d", len(cleaned), limit)

        response = self._client.get(
            "https://www.reddit.com/search.json",
            params={"q": cleaned, "limit": limit, "sort": "relevance", "type": "link"},
            headers={"User-Agent": self._user_agent},
        )
        raise_for_status(response, provider="reddit")
        children = (response.json().get("data") or {}).get("children") or []

        documents: list[Document] = []
        for child in children[:limit]:
            if not isinstance(child, dict):
                continue
            data = child.get("data") or {}
            if not isinstance(data, dict):
                continue
            title = str(data.get("title") or "").strip()
            permalink = str(data.get("permalink") or "").strip()
            external_url = str(data.get("url") or "").strip()
            url = (
                f"https://www.reddit.com{permalink}" if permalink.startswith("/") else external_url
            )
            if not title or not url:
                continue
            body = str(data.get("selftext") or "").strip()
            documents.append(
                Document(
                    title=title,
                    url=url,
                    snippet=(body or title)[:400],
                    content=body,
                    provider=self.name,
                    metadata={
                        "subreddit": data.get("subreddit"),
                        "score": data.get("score"),
                        "num_comments": data.get("num_comments"),
                    },
                )
            )

        logger.info("reddit returned count=%d", len(documents))
        return documents
