"""Temporary in-memory vector store (no external dependencies)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from insightforge.core.logging import get_logger
from insightforge.domain.models import VectorHit, VectorRecord
from insightforge.infrastructure.vectorstores.base import VectorStore
from insightforge.infrastructure.vectorstores.helpers import (
    cosine_similarity,
    matches_filters,
    require_dimensions,
    require_records,
    require_vector,
)
from insightforge.shared.enums import VectorStoreHint

logger = get_logger(__name__)


class MemoryVectorStore(VectorStore):
    """Process-local temporary vector index using cosine similarity.

    Suitable for unit tests, short-lived research sessions, and local work
    without Qdrant. Not shared across processes.
    """

    name = VectorStoreHint.MEMORY

    def __init__(self, *, dimensions: int | None = None) -> None:
        self._dimensions = dimensions
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        cleaned = require_records(records)
        if not cleaned:
            return 0

        dims = require_dimensions(
            [record.vector for record in cleaned],
            expected=self._dimensions,
        )
        if self._dimensions is None:
            self._dimensions = dims
            logger.info(
                "memory store dimensions inferred dims=%d",
                dims,
                extra={"store": self.name.value, "dimensions": dims},
            )

        for record in cleaned:
            self._records[record.id] = record.model_copy(deep=True)

        logger.info(
            "memory upsert count=%d total=%d",
            len(cleaned),
            len(self._records),
            extra={"store": self.name.value, "upserted": len(cleaned), "total": len(self._records)},
        )
        return len(cleaned)

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        with_vectors: bool = False,
    ) -> list[VectorHit]:
        query = require_vector(vector)
        if self._dimensions is not None:
            require_dimensions([query], expected=self._dimensions)
        if limit < 1:
            return []

        hits: list[VectorHit] = []
        for record in self._records.values():
            if not matches_filters(record.metadata, filters):
                continue
            score = cosine_similarity(query, record.vector)
            if score_threshold is not None and score < score_threshold:
                continue
            hits.append(
                VectorHit(
                    id=record.id,
                    score=score,
                    text=record.text,
                    metadata=dict(record.metadata),
                    vector=list(record.vector) if with_vectors else None,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        results = hits[:limit]
        logger.debug(
            "memory search hits=%d limit=%d",
            len(results),
            limit,
            extra={"store": self.name.value, "hits": len(results)},
        )
        return results

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> int:
        if ids is None and filters is None:
            return 0

        id_set = {item for item in (ids or []) if item}
        to_delete: list[str] = []
        for record_id, record in self._records.items():
            id_ok = True if ids is None else record_id in id_set
            filter_ok = matches_filters(record.metadata, filters)
            if id_ok and filter_ok:
                to_delete.append(record_id)

        for record_id in to_delete:
            del self._records[record_id]

        logger.info(
            "memory delete removed=%d remaining=%d",
            len(to_delete),
            len(self._records),
            extra={"store": self.name.value, "removed": len(to_delete)},
        )
        return len(to_delete)

    def count(self, *, filters: Mapping[str, Any] | None = None) -> int:
        if not filters:
            return len(self._records)
        return sum(
            1 for record in self._records.values() if matches_filters(record.metadata, filters)
        )

    def clear(self) -> None:
        size = len(self._records)
        self._records.clear()
        logger.info(
            "memory store cleared previous=%d",
            size,
            extra={"store": self.name.value, "previous": size},
        )
