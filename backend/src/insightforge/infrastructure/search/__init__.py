"""Search infrastructure — providers that gather documents for research tasks."""

from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.dedupe import dedupe_documents, normalize_url
from insightforge.infrastructure.search.http import close_http_client, create_http_client
from insightforge.infrastructure.search.providers import (
    ArxivSearchProvider,
    GitHubSearchProvider,
    RedditSearchProvider,
    WebSearchProvider,
    WikipediaSearchProvider,
    YouTubeSearchProvider,
)
from insightforge.infrastructure.search.rate_limit import RateLimiter
from insightforge.infrastructure.search.scoring import (
    provider_metadata_score,
    score_document,
    score_documents,
)
from insightforge.infrastructure.search.service import (
    SearchService,
    build_providers,
    create_search_service,
)

__all__ = [
    "ArxivSearchProvider",
    "GitHubSearchProvider",
    "RateLimiter",
    "RedditSearchProvider",
    "SearchProvider",
    "SearchService",
    "WebSearchProvider",
    "WikipediaSearchProvider",
    "YouTubeSearchProvider",
    "build_providers",
    "close_http_client",
    "create_http_client",
    "create_search_service",
    "dedupe_documents",
    "normalize_url",
    "provider_metadata_score",
    "score_document",
    "score_documents",
]
