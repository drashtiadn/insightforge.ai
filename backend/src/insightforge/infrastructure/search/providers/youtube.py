"""YouTube video search via the Data API v3."""

from __future__ import annotations

import httpx

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.http import raise_for_status, require_query
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)


class YouTubeSearchProvider(SearchProvider):
    """Search YouTube videos. Requires ``YOUTUBE_API_KEY``."""

    name = SearchProviderHint.YOUTUBE

    def __init__(
        self,
        client: httpx.Client,
        *,
        api_key: str | None = None,
    ) -> None:
        self._client = client
        self._api_key = (api_key or "").strip() or None

    @property
    def available(self) -> bool:
        return self._api_key is not None

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        cleaned = require_query(query)
        if not self._api_key:
            logger.warning("youtube search skipped: YOUTUBE_API_KEY not configured")
            return []

        logger.info("youtube search query_len=%d limit=%d", len(cleaned), limit)
        response = self._client.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "type": "video",
                "q": cleaned,
                "maxResults": limit,
                "key": self._api_key,
            },
        )
        raise_for_status(response, provider="youtube")
        items = response.json().get("items") or []

        documents: list[Document] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            video_id = (item.get("id") or {}).get("videoId")
            snippet = item.get("snippet") or {}
            if not isinstance(snippet, dict) or not video_id:
                continue
            title = str(snippet.get("title") or "").strip()
            description = str(snippet.get("description") or "").strip()
            if not title:
                continue
            documents.append(
                Document(
                    title=title,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    snippet=description[:400],
                    content=description,
                    provider=self.name,
                    metadata={
                        "video_id": video_id,
                        "channel": snippet.get("channelTitle"),
                        "published_at": snippet.get("publishedAt"),
                    },
                )
            )

        logger.info("youtube returned count=%d", len(documents))
        return documents
