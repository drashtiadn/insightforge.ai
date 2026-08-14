"""Reranker provider contract shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from insightforge.domain.models import RetrievalHit
from insightforge.shared.enums import RerankerHint


class RerankerProvider(ABC):
    """One query-document reranker.

    Empty candidate lists return empty results without calling a model.
    Transport and API failures should raise ``ExternalServiceError``.
    """

    name: RerankerHint

    @property
    def available(self) -> bool:
        """True when required config is present and the provider can run."""

        return True

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier used for rerank calls."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        """Return ``hits`` ordered by relevance to ``query`` (highest first)."""
