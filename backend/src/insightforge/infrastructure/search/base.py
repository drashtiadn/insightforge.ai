"""Search provider contract shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from insightforge.domain.models import Document
from insightforge.shared.enums import SearchProviderHint


class SearchProvider(ABC):
    """One external search backend.

    Providers must not raise for empty result sets. Transport and API failures
    should raise ``ExternalServiceError`` so the service layer can soft-fail
    per provider.
    """

    name: SearchProviderHint

    @property
    def available(self) -> bool:
        """True when required config is present and the provider can run."""

        return True

    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        """Return up to ``limit`` documents for ``query``."""
