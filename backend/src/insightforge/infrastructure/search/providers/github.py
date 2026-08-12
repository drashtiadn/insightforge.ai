"""GitHub repository search."""

from __future__ import annotations

import httpx

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.http import raise_for_status, require_query
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)


class GitHubSearchProvider(SearchProvider):
    """Search public GitHub repositories."""

    name = SearchProviderHint.GITHUB

    def __init__(
        self,
        client: httpx.Client,
        *,
        token: str | None = None,
    ) -> None:
        self._client = client
        self._token = (token or "").strip() or None

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        cleaned = require_query(query)
        logger.info("github search query_len=%d limit=%d", len(cleaned), limit)

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        response = self._client.get(
            "https://api.github.com/search/repositories",
            params={"q": cleaned, "per_page": limit, "sort": "stars", "order": "desc"},
            headers=headers,
        )
        raise_for_status(response, provider="github")
        items = response.json().get("items") or []

        documents: list[Document] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("full_name") or "").strip()
            url = str(item.get("html_url") or "").strip()
            if not title or not url:
                continue
            description = str(item.get("description") or "").strip()
            documents.append(
                Document(
                    title=title,
                    url=url,
                    snippet=description[:400],
                    content=description,
                    provider=self.name,
                    metadata={
                        "stars": item.get("stargazers_count"),
                        "language": item.get("language"),
                        "forks": item.get("forks_count"),
                    },
                )
            )

        logger.info("github returned count=%d", len(documents))
        return documents
