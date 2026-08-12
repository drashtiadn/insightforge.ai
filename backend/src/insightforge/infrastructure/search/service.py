"""Search service — parallel providers with rate limits, dedupe, and scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import TracebackType
from typing import Self

import httpx

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document, ResearchTask
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.dedupe import dedupe_documents
from insightforge.infrastructure.search.http import (
    close_http_client,
    create_http_client,
    require_query,
)
from insightforge.infrastructure.search.providers import (
    ArxivSearchProvider,
    GitHubSearchProvider,
    RedditSearchProvider,
    WebSearchProvider,
    WikipediaSearchProvider,
    YouTubeSearchProvider,
)
from insightforge.infrastructure.search.rate_limit import RateLimiter
from insightforge.infrastructure.search.scoring import score_documents
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
    """Execute searches across providers with a production-ready pipeline.

    Pipeline: rate-limit → parallel fetch → dedupe → score → trim.
    Provider failures are logged and skipped so one outage does not abort the run.

    When constructed via ``create_search_service`` without an injected client,
    this service owns the shared ``httpx.Client`` and closes it on ``close()``
    or context-manager exit.
    """

    def __init__(
        self,
        providers: Mapping[SearchProviderHint, SearchProvider],
        *,
        default_limit: int = 5,
        max_workers: int = 4,
        rate_limiter: RateLimiter | None = None,
        dedupe: bool = True,
        scoring: bool = True,
        max_documents: int = 20,
        client: httpx.Client | None = None,
        owns_client: bool = False,
    ) -> None:
        self._providers = dict(providers)
        self._default_limit = default_limit
        self._max_workers = max(1, max_workers)
        self._rate_limiter = rate_limiter
        self._dedupe = dedupe
        self._scoring = scoring
        self._max_documents = max(1, max_documents)
        self._client = client
        self._owns_client = owns_client and client is not None

    @property
    def providers(self) -> dict[SearchProviderHint, SearchProvider]:
        return dict(self._providers)

    @property
    def client(self) -> httpx.Client | None:
        """Shared HTTP client when one is managed by this service."""

        return self._client

    @property
    def owns_client(self) -> bool:
        return self._owns_client

    def get_provider(self, hint: SearchProviderHint) -> SearchProvider | None:
        return self._providers.get(hint)

    def close(self) -> None:
        """Close the owned HTTP client, if any. Idempotent."""

        if not self._owns_client:
            return
        close_http_client(self._client)
        self._client = None
        self._owns_client = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def search(
        self,
        query: str,
        providers: Sequence[SearchProviderHint],
        *,
        limit: int | None = None,
    ) -> list[Document]:
        """Run ``query`` against providers in parallel, then post-process."""

        cleaned = require_query(query)
        if not providers:
            raise ValidationFailedError(
                "at least one search provider is required",
                details={"field": "providers"},
            )

        max_results = limit if limit is not None else self._default_limit
        hints = list(dict.fromkeys(providers))  # preserve order, drop dup hints

        logger.info(
            "search started query_len=%d providers=%s limit=%d workers=%d",
            len(cleaned),
            [hint.value for hint in hints],
            max_results,
            self._max_workers,
        )

        documents = self._fetch_parallel(cleaned, hints, max_results)
        before = len(documents)

        if self._dedupe:
            documents = dedupe_documents(documents)

        if self._scoring:
            documents = score_documents(documents, cleaned)
        elif self._dedupe:
            # Still prefer richer docs first when scoring is off.
            documents = sorted(
                documents,
                key=lambda doc: len(doc.snippet) + len(doc.content),
                reverse=True,
            )

        if len(documents) > self._max_documents:
            logger.info(
                "trimming documents from=%d to=%d",
                len(documents),
                self._max_documents,
            )
            documents = documents[: self._max_documents]

        logger.info(
            "search finished raw=%d final=%d dedupe=%s scoring=%s",
            before,
            len(documents),
            self._dedupe,
            self._scoring,
            extra={"raw_count": before, "final_count": len(documents)},
        )
        return documents

    def search_task(
        self,
        task: ResearchTask,
        *,
        limit: int | None = None,
    ) -> list[Document]:
        """Search using a domain ``ResearchTask``."""

        logger.info(
            "search task id=%s priority=%d providers=%s",
            task.id,
            task.priority,
            [hint.value for hint in task.providers],
            extra={"task_id": task.id, "priority": task.priority},
        )
        return self.search(task.search_query, task.providers, limit=limit)

    def search_tasks(
        self,
        tasks: Sequence[ResearchTask],
        *,
        limit: int | None = None,
    ) -> list[Document]:
        """Run multiple research tasks, scoring each against its own query."""

        if not tasks:
            return []

        max_results = limit if limit is not None else self._default_limit
        ordered = sorted(tasks, key=lambda task: task.priority)
        merged: list[Document] = []

        for task in ordered:
            query = require_query(task.search_query)
            if not task.providers:
                logger.warning("search task skipped empty providers task_id=%s", task.id)
                continue
            hints = list(dict.fromkeys(task.providers))
            logger.info(
                "search task id=%s priority=%d providers=%s",
                task.id,
                task.priority,
                [hint.value for hint in hints],
                extra={"task_id": task.id, "priority": task.priority},
            )
            hits = self._fetch_parallel(query, hints, max_results)
            if self._scoring:
                hits = score_documents(hits, query)
            merged.extend(hits)

        before = len(merged)
        if self._dedupe:
            merged = dedupe_documents(merged)
            # Dedupe may reorder; prefer higher application scores when present.
            merged = sorted(
                merged,
                key=lambda doc: (
                    doc.score if doc.score is not None else -1.0,
                    len(doc.snippet) + len(doc.content),
                ),
                reverse=True,
            )
        elif self._scoring:
            merged = sorted(
                merged,
                key=lambda doc: doc.score or 0.0,
                reverse=True,
            )

        if len(merged) > self._max_documents:
            merged = merged[: self._max_documents]

        logger.info(
            "search_tasks finished tasks=%d raw=%d final=%d",
            len(ordered),
            before,
            len(merged),
        )
        return merged

    def _fetch_parallel(
        self,
        query: str,
        hints: Sequence[SearchProviderHint],
        limit: int,
    ) -> list[Document]:
        runnable: list[SearchProviderHint] = []
        for hint in hints:
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
            runnable.append(hint)

        if not runnable:
            return []

        documents: list[Document] = []
        workers = min(self._max_workers, len(runnable))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._search_one, hint, query, limit): hint for hint in runnable}
            for future in as_completed(futures):
                hint = futures[future]
                try:
                    hits = future.result()
                except Exception as exc:
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
        return documents

    def _search_one(
        self,
        hint: SearchProviderHint,
        query: str,
        limit: int,
    ) -> list[Document]:
        provider = self._providers[hint]
        if self._rate_limiter is not None:
            self._rate_limiter.acquire(hint.value)
        try:
            return provider.search(query, limit=limit)
        except (ExternalServiceError, ValidationFailedError, httpx.HTTPError):
            raise


def create_search_service(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
) -> SearchService:
    """Factory used by application code and tests.

    When ``client`` is omitted, the service owns the created HTTP client and
    callers should use ``close()`` or a ``with`` block to release it.
    """

    cfg = settings or get_settings()
    owns_client = client is None
    http_client = client or create_http_client(timeout=cfg.search_timeout_seconds)
    limiter = RateLimiter(calls_per_second=cfg.search_rate_limit_per_second)
    return SearchService(
        build_providers(cfg, http_client),
        default_limit=cfg.search_default_limit,
        max_workers=cfg.search_max_workers,
        rate_limiter=limiter,
        dedupe=cfg.search_dedupe_enabled,
        scoring=cfg.search_scoring_enabled,
        max_documents=cfg.search_max_documents,
        client=http_client,
        owns_client=owns_client,
    )
