"""Search infrastructure — providers that gather documents for research tasks."""

from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.providers import (
    ArxivSearchProvider,
    GitHubSearchProvider,
    RedditSearchProvider,
    WebSearchProvider,
    WikipediaSearchProvider,
    YouTubeSearchProvider,
)
from insightforge.infrastructure.search.service import (
    SearchService,
    build_providers,
    create_search_service,
)

__all__ = [
    "ArxivSearchProvider",
    "GitHubSearchProvider",
    "RedditSearchProvider",
    "SearchProvider",
    "SearchService",
    "WebSearchProvider",
    "WikipediaSearchProvider",
    "YouTubeSearchProvider",
    "build_providers",
    "create_search_service",
]
