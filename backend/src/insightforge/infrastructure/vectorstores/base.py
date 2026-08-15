"""Vector store contract shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from insightforge.domain.models import VectorHit, VectorRecord
from insightforge.shared.enums import VectorStoreHint


class VectorStore(ABC):
    """Persistent or ephemeral dense-vector index.

    Empty upsert/search inputs are no-ops / empty results. Transport failures
    should raise ``ExternalServiceError``. Invalid arguments raise
    ``ValidationFailedError``.
    """

    name: VectorStoreHint

    @property
    def available(self) -> bool:
        """True when the store can accept reads and writes."""

        return True

    @property
    @abstractmethod
    def dimensions(self) -> int | None:
        """Configured vector size, or ``None`` until inferred on first upsert."""

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace records. Returns the number written."""

    @abstractmethod
    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        with_vectors: bool = False,
    ) -> list[VectorHit]:
        """Return nearest neighbors for ``vector``, highest score first."""

    @abstractmethod
    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> int:
        """Delete by ids and/or metadata equality filters. Returns count removed."""

    @abstractmethod
    def count(self, *, filters: Mapping[str, Any] | None = None) -> int:
        """Count stored points, optionally filtered by metadata equality."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all points from this store/collection."""
