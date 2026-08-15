"""Reranker agent — reorder retrieved chunks by query relevance."""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.agents.base import BaseAgent
from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit
from insightforge.infrastructure.rerankers import RerankerService

logger = get_logger(__name__)


class SimpleReranker(BaseAgent):
    """Thin agent over ``RerankerService`` (no provider SDKs)."""

    name = "reranker"

    def __init__(self, reranker: RerankerService) -> None:
        super().__init__()
        self._reranker = reranker

    def run(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        """Return ``hits`` ordered by relevance to ``query``."""

        logger.info(
            "reranker run query_len=%d candidates=%d top_n=%s",
            len(query),
            len(hits),
            top_n,
        )
        return self._reranker.rerank(query, hits, top_n=top_n)
