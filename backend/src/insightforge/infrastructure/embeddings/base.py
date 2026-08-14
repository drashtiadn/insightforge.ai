"""Embedding provider contract shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from insightforge.domain.models import EmbeddingResult
from insightforge.shared.enums import EmbeddingInputType, EmbeddingProviderHint


class EmbeddingProvider(ABC):
    """One text-embedding backend.

    Empty input lists must return an empty result without calling the network
    or loading a model. Transport and API failures should raise
    ``ExternalServiceError`` so callers can decide how to soft-fail.
    """

    name: EmbeddingProviderHint

    @property
    def available(self) -> bool:
        """True when required config is present and the provider can run."""

        return True

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier used for embedding calls."""

    @abstractmethod
    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> EmbeddingResult:
        """Embed ``texts`` and return dense vectors in input order."""
