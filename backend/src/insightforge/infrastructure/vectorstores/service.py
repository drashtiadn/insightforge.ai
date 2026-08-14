"""Vector store service — primary backend plus session-scoped temp stores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import VectorHit, VectorRecord
from insightforge.infrastructure.vectorstores.base import VectorStore
from insightforge.infrastructure.vectorstores.session import SessionVectorStoreRegistry
from insightforge.infrastructure.vectorstores.stores import MemoryVectorStore, QdrantVectorStore
from insightforge.shared.enums import VectorStoreHint

logger = get_logger(__name__)


def parse_vector_store(value: str) -> VectorStoreHint:
    """Parse settings/env value into ``VectorStoreHint``."""

    cleaned = value.strip().lower()
    try:
        return VectorStoreHint(cleaned)
    except ValueError as exc:
        allowed = ", ".join(hint.value for hint in VectorStoreHint)
        raise ValidationFailedError(
            f"unknown vector store {value!r}; expected one of: {allowed}",
            details={"field": "vector_store", "value": value},
        ) from exc


def build_store(
    settings: Settings,
    *,
    qdrant_client: Any | None = None,
) -> VectorStore:
    """Construct the configured primary vector store."""

    hint = parse_vector_store(settings.vector_store)
    if hint is VectorStoreHint.MEMORY:
        return MemoryVectorStore(dimensions=settings.vector_dimensions)

    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=api_key,
        collection=settings.qdrant_collection,
        dimensions=settings.vector_dimensions,
        timeout=settings.qdrant_timeout_seconds,
        client=qdrant_client,
    )


class VectorStoreService:
    """Facade over the primary store and per-session temporary indexes.

    - ``upsert`` / ``search`` / ``delete`` / ``count`` / ``clear`` target the
      primary backend (memory or Qdrant).
    - ``session(session_id)`` returns an isolated temporary ``MemoryVectorStore``
      for one research session (TTL-managed).
    """

    def __init__(
        self,
        store: VectorStore,
        *,
        sessions: SessionVectorStoreRegistry | None = None,
    ) -> None:
        self._store = store
        self._sessions = sessions or SessionVectorStoreRegistry()

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def sessions(self) -> SessionVectorStoreRegistry:
        return self._sessions

    @property
    def available(self) -> bool:
        return self._store.available

    @property
    def dimensions(self) -> int | None:
        return self._store.dimensions

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        self._require_available()
        return self._store.upsert(records)

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        with_vectors: bool = False,
    ) -> list[VectorHit]:
        self._require_available()
        return self._store.search(
            vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
            with_vectors=with_vectors,
        )

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> int:
        self._require_available()
        return self._store.delete(ids=ids, filters=filters)

    def count(self, *, filters: Mapping[str, Any] | None = None) -> int:
        self._require_available()
        return self._store.count(filters=filters)

    def clear(self) -> None:
        self._require_available()
        self._store.clear()

    def session(self, session_id: str) -> MemoryVectorStore:
        """Temporary per-session vector store (isolated from the primary index)."""

        return self._sessions.get(session_id)

    def drop_session(self, session_id: str) -> bool:
        """Drop one session store."""

        return self._sessions.drop(session_id)

    def cleanup_sessions(self) -> int:
        """Expire idle session stores. Returns how many were removed."""

        return self._sessions.cleanup()

    def _require_available(self) -> None:
        if not self._store.available:
            raise ExternalServiceError(
                f"vector store unavailable: {self._store.name.value}",
                details={"store": self._store.name.value},
            )


def create_vector_store_service(
    settings: Settings | None = None,
    *,
    qdrant_client: Any | None = None,
) -> VectorStoreService:
    """Factory used by application code and tests."""

    cfg = settings or get_settings()
    store = build_store(cfg, qdrant_client=qdrant_client)
    sessions = SessionVectorStoreRegistry(
        ttl_seconds=cfg.vector_session_ttl_seconds,
        dimensions=cfg.vector_dimensions,
    )
    logger.info(
        "vector store service created store=%s dims=%s available=%s",
        store.name.value,
        store.dimensions,
        store.available,
        extra={
            "store": store.name.value,
            "dimensions": store.dimensions,
            "available": store.available,
        },
    )
    return VectorStoreService(store, sessions=sessions)
