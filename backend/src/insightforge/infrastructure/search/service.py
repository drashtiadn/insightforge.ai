"""Search service — routes planner tasks to concrete providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import httpx

from insightforge.agents.planner.schemas import ResearchTask
from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.http import create_http_client, require_query
from insightforge.infrastructure.search.providers import (
    ArxivSearchProvider,
    GitHubSearchProvider,
    RedditSearchProvider,
    WebSearchProvider,
    WikipediaSearchProvider,
    YouTubeSearchProvider,
)
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)


def build_providers(
    settings: Settings,
    client: httpx.Client,
) -> dict[SearchProviderHint, SearchProvider]:
    """Construct the default provider map from settings."""

    tavily = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
    github = settings.github_token.get_secret_value() if settings.github_token else None
    youtube = settings.youtube_api_key.get_secret_value() if settings.youtube_api_key else None

    return {
        SearchProviderHint.WEB: WebSearchProvider(
            client,
            tavily_api_key=tavily,
            searxng_base_url=settings.searxng_base_url,
        ),
        SearchProviderHint.GITHUB: GitHubSearchProvider(client, token=github),
        SearchProviderHint.ARXIV: ArxivSearchProvider(client),
        SearchProviderHint.WIKIPEDIA: WikipediaSearchProvider(client),
        SearchProviderHint.YOUTUBE: YouTubeSearchProvider(client, api_key=youtube),
        SearchProviderHint.REDDIT: RedditSearchProvider(
            client,
            user_agent=settings.reddit_user_agent,
        ),
    }


class SearchService:
    """Execute searches across one or more providers.

    Provider failures are logged and skipped so a single outage does not abort
    the whole research step. Parallelism / dedupe / scoring land in Phase 3.3.
    """

    def __init__(
        self,
        providers: Mapping[SearchProviderHint, SearchProvider],
        *,
        default_limit: int = 5,
    ) -> None:
        self._providers = dict(providers)
        self._default_limit = default_limit

    @property
    def providers(self) -> dict[SearchProviderHint, SearchProvider]:
        return dict(self._providers)

    def get_provider(self, hint: SearchProviderHint) -> SearchProvider | None:
        return self._providers.get(hint)

    def search(
        self,
        query: str,
        providers: Sequence[SearchProviderHint],
        *,
        limit: int | None = None,
    ) -> list[Document]:
        """Run ``query`` against each requested provider (sequentially)."""

        cleaned = require_query(query)
        if not providers:
            raise ValidationFailedError(
                "at least one search provider is required",
                details={"field": "providers"},
            )

        max_results = limit if limit is not None else self._default_limit
        documents: list[Document] = []

        logger.info(
            "search started query_len=%d providers=%s limit=%d",
            len(cleaned),
            [hint.value for hint in providers],
            max_results,
        )

        for hint in providers:
            provider = self._providers.get(hint)
            if provider is None:
                logger.warning("search provider not registered provider=%s", hint.value)
                continue
            if not provider.available:
                logger.warning(
                    "search provider unavailable provider=%s",
                    hint.value,
                    extra={"provider": hint.value},
                )
                continue
            try:
                hits = provider.search(cleaned, limit=max_results)
            except (ExternalServiceError, ValidationFailedError, httpx.HTTPError) as exc:
                logger.warning(
                    "search provider failed provider=%s error=%s",
                    hint.value,
                    exc,
                    extra={"provider": hint.value},
                )
                continue
            logger.debug(
                "search provider ok provider=%s count=%d",
                hint.value,
                len(hits),
            )
            documents.extend(hits)

        logger.info("search finished document_count=%d", len(documents))
        return documents

    def search_task(
        self,
        task: ResearchTask,
        *,
        limit: int | None = None,
    ) -> list[Document]:
        """Search using a planner ``ResearchTask``."""

        logger.info(
            "search task id=%s priority=%d providers=%s",
            task.id,
            task.priority,
            [hint.value for hint in task.providers],
            extra={"task_id": task.id, "priority": task.priority},
        )
        return self.search(task.search_query, task.providers, limit=limit)


def create_search_service(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
) -> SearchService:
    """Factory used by application code and tests."""

    cfg = settings or get_settings()
    http_client = client or create_http_client(timeout=cfg.search_timeout_seconds)
    return SearchService(
        build_providers(cfg, http_client),
        default_limit=cfg.search_default_limit,
    )
