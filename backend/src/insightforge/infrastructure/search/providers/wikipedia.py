"""Wikipedia search via the MediaWiki API."""

from __future__ import annotations

import re
from urllib.parse import quote

import httpx

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.http import raise_for_status, require_query
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(value: str) -> str:
    return _TAG_RE.sub("", value).strip()


class WikipediaSearchProvider(SearchProvider):
    """Search English Wikipedia articles."""

    name = SearchProviderHint.WIKIPEDIA

    def __init__(
        self,
        client: httpx.Client,
        *,
        language: str = "en",
    ) -> None:
        self._client = client
        self._language = language

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        cleaned = require_query(query)
        logger.info("wikipedia search query_len=%d limit=%d", len(cleaned), limit)

        response = self._client.get(
            f"https://{self._language}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": cleaned,
                "srlimit": limit,
                "format": "json",
                "utf8": 1,
            },
            headers={"User-Agent": "insightforge/0.1.0 (research platform)"},
        )
        raise_for_status(response, provider="wikipedia")
        results = (response.json().get("query") or {}).get("search") or []

        documents: list[Document] = []
        for item in results[:limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            snippet = _strip_html(str(item.get("snippet") or ""))
            url = f"https://{self._language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            documents.append(
                Document(
                    title=title,
                    url=url,
                    snippet=snippet[:400],
                    content=snippet,
                    provider=self.name,
                    metadata={"pageid": item.get("pageid")},
                )
            )

        logger.info("wikipedia returned count=%d", len(documents))
        return documents
