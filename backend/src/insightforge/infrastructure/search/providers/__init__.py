"""Concrete search provider implementations."""

from insightforge.infrastructure.search.providers.arxiv import ArxivSearchProvider
from insightforge.infrastructure.search.providers.github import GitHubSearchProvider
from insightforge.infrastructure.search.providers.reddit import RedditSearchProvider
from insightforge.infrastructure.search.providers.web import WebSearchProvider
from insightforge.infrastructure.search.providers.wikipedia import WikipediaSearchProvider
from insightforge.infrastructure.search.providers.youtube import YouTubeSearchProvider

__all__ = [
    "ArxivSearchProvider",
    "GitHubSearchProvider",
    "RedditSearchProvider",
    "WebSearchProvider",
    "WikipediaSearchProvider",
    "YouTubeSearchProvider",
]
