"""Qdrant vector store backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid5

from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import VectorHit, VectorRecord
from insightforge.infrastructure.vectorstores.base import VectorStore
from insightforge.infrastructure.vectorstores.helpers import (
    require_dimensions,
    require_records,
    require_vector,
)
from insightforge.shared.enums import VectorStoreHint

logger = get_logger(__name__)

# Stable namespace so non-UUID record ids map to deterministic Qdrant point ids.
_POINT_NAMESPACE = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace
_PAYLOAD_ID_KEY = "record_id"
_PAYLOAD_TEXT_KEY = "text"


def point_id_for(record_id: str) -> str:
    """Map an application record id to a Qdrant-compatible UUID string."""

    try:
        return str(UUID(record_id))
    except ValueError:
        return str(uuid5(_POINT_NAMESPACE, record_id))


def build_payload(record: VectorRecord) -> dict[str, Any]:
    """Flatten record fields into a Qdrant payload."""

    payload = dict(record.metadata)
    payload[_PAYLOAD_ID_KEY] = record.id
    payload[_PAYLOAD_TEXT_KEY] = record.text
    return payload


def payload_to_hit(
    *,
    point_id: str | int,
    score: float,
    payload: Mapping[str, Any] | None,
    vector: Sequence[float] | None,
) -> VectorHit:
    """Convert a Qdrant scored point into a domain ``VectorHit``."""

    data = dict(payload or {})
    record_id = str(data.pop(_PAYLOAD_ID_KEY, point_id))
    text = str(data.pop(_PAYLOAD_TEXT_KEY, "") or "")
    return VectorHit(
        id=record_id,
        score=float(score),
        text=text,
        metadata=data,
        vector=[float(value) for value in vector] if vector is not None else None,
    )


class QdrantVectorStore(VectorStore):
    """Remote Qdrant collection (inject ``client`` in tests)."""

    name = VectorStoreHint.QDRANT

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = "insightforge",
        dimensions: int = 1024,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._url = (url or "").strip().rstrip("/") or "http://localhost:6333"
        self._api_key = (api_key or "").strip() or None
        self._collection = (collection or "").strip() or "insightforge"
        if dimensions < 1:
            raise ValidationFailedError(
                "vector dimensions must be >= 1",
                details={"field": "dimensions", "value": dimensions},
            )
        self._dimensions = dimensions
        self._timeout = timeout
        self._client = client
        self._ensured = False

    @property
    def available(self) -> bool:
        return True

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def collection(self) -> str:
        return self._collection

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        cleaned = require_records(records)
        if not cleaned:
            return 0

        require_dimensions([record.vector for record in cleaned], expected=self._dimensions)
        client = self._get_client()
        self._ensure_collection(client)

        from qdrant_client.http import models as qmodels

        points = [
            qmodels.PointStruct(
                id=point_id_for(record.id),
                vector=list(record.vector),
                payload=build_payload(record),
            )
            for record in cleaned
        ]
        try:
            client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:
            logger.warning(
                "qdrant upsert failed collection=%s error=%s",
                self._collection,
                exc,
                extra={"store": self.name.value, "collection": self._collection},
            )
            raise ExternalServiceError(
                f"qdrant upsert failed: {exc}",
                details={"store": self.name.value, "collection": self._collection},
            ) from exc

        logger.info(
            "qdrant upsert count=%d collection=%s",
            len(points),
            self._collection,
            extra={"store": self.name.value, "upserted": len(points)},
        )
        return len(points)

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
        require_dimensions([query], expected=self._dimensions)
        if limit < 1:
            return []

        client = self._get_client()
        self._ensure_collection(client)
        query_filter = self._build_filter(filters)

        try:
            response = client.query_points(
                collection_name=self._collection,
                query=query,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as exc:
            logger.warning(
                "qdrant search failed collection=%s error=%s",
                self._collection,
                exc,
                extra={"store": self.name.value, "collection": self._collection},
            )
            raise ExternalServiceError(
                f"qdrant search failed: {exc}",
                details={"store": self.name.value, "collection": self._collection},
            ) from exc

        points = getattr(response, "points", response) or []
        hits: list[VectorHit] = []
        for point in points:
            raw_vector = None
            if with_vectors:
                raw_vector = getattr(point, "vector", None)
                if isinstance(raw_vector, dict):
                    raw_vector = next(iter(raw_vector.values()), None)
            hits.append(
                payload_to_hit(
                    point_id=getattr(point, "id", ""),
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    payload=getattr(point, "payload", None),
                    vector=raw_vector if isinstance(raw_vector, list | tuple) else None,
                )
            )

        logger.debug(
            "qdrant search hits=%d limit=%d collection=%s",
            len(hits),
            limit,
            self._collection,
            extra={"store": self.name.value, "hits": len(hits)},
        )
        return hits

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> int:
        if ids is None and filters is None:
            return 0

        client = self._get_client()
        self._ensure_collection(client)

        from qdrant_client.http import models as qmodels

        try:
            if ids is not None and filters is None:
                point_ids = [point_id_for(item) for item in ids if item]
                if not point_ids:
                    return 0
                client.delete(
                    collection_name=self._collection,
                    points_selector=qmodels.PointIdsList(points=point_ids),
                    wait=True,
                )
                removed = len(point_ids)
            else:
                conditions: list[Any] = []
                if ids is not None:
                    record_ids = [item for item in ids if item]
                    if not record_ids:
                        return 0
                    conditions.append(
                        qmodels.FieldCondition(
                            key=_PAYLOAD_ID_KEY,
                            match=qmodels.MatchAny(any=record_ids),
                        )
                    )
                if filters:
                    conditions.extend(self._filter_conditions(filters))

                before = self.count()
                client.delete(
                    collection_name=self._collection,
                    points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=conditions)),
                    wait=True,
                )
                removed = max(0, before - self.count())
        except ExternalServiceError:
            raise
        except Exception as exc:
            logger.warning(
                "qdrant delete failed collection=%s error=%s",
                self._collection,
                exc,
                extra={"store": self.name.value, "collection": self._collection},
            )
            raise ExternalServiceError(
                f"qdrant delete failed: {exc}",
                details={"store": self.name.value, "collection": self._collection},
            ) from exc

        logger.info(
            "qdrant delete removed=%d collection=%s",
            removed,
            self._collection,
            extra={"store": self.name.value, "removed": removed},
        )
        return int(removed)

    def count(self, *, filters: Mapping[str, Any] | None = None) -> int:
        client = self._get_client()
        self._ensure_collection(client)
        query_filter = self._build_filter(filters)
        try:
            result = client.count(
                collection_name=self._collection,
                count_filter=query_filter,
                exact=True,
            )
        except Exception as exc:
            raise ExternalServiceError(
                f"qdrant count failed: {exc}",
                details={"store": self.name.value, "collection": self._collection},
            ) from exc
        return int(getattr(result, "count", 0) or 0)

    def clear(self) -> None:
        client = self._get_client()
        try:
            if (
                hasattr(client, "collection_exists") and client.collection_exists(self._collection)
            ) or not hasattr(client, "collection_exists"):
                client.delete_collection(self._collection)
        except Exception as exc:
            message = str(exc).lower()
            if "not found" not in message and "doesn't exist" not in message:
                raise ExternalServiceError(
                    f"qdrant clear failed: {exc}",
                    details={"store": self.name.value, "collection": self._collection},
                ) from exc
        self._ensured = False
        logger.info(
            "qdrant collection cleared collection=%s",
            self._collection,
            extra={"store": self.name.value, "collection": self._collection},
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient

        logger.info(
            "connecting qdrant url=%s collection=%s",
            self._url,
            self._collection,
            extra={"store": self.name.value, "url": self._url},
        )
        self._client = QdrantClient(
            url=self._url,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        return self._client

    def _ensure_collection(self, client: Any) -> None:
        if self._ensured:
            return
        from qdrant_client.http import models as qmodels

        exists = False
        try:
            if hasattr(client, "collection_exists"):
                exists = bool(client.collection_exists(self._collection))
            else:
                client.get_collection(self._collection)
                exists = True
        except Exception:
            exists = False

        if not exists:
            logger.info(
                "creating qdrant collection collection=%s dims=%d",
                self._collection,
                self._dimensions,
                extra={
                    "store": self.name.value,
                    "collection": self._collection,
                    "dimensions": self._dimensions,
                },
            )
            try:
                client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=self._dimensions,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
            except Exception as exc:
                raise ExternalServiceError(
                    f"qdrant create_collection failed: {exc}",
                    details={"store": self.name.value, "collection": self._collection},
                ) from exc
        self._ensured = True

    def _build_filter(self, filters: Mapping[str, Any] | None) -> Any | None:
        if not filters:
            return None
        from qdrant_client.http import models as qmodels

        return qmodels.Filter(must=self._filter_conditions(filters))

    def _filter_conditions(self, filters: Mapping[str, Any]) -> list[Any]:
        from qdrant_client.http import models as qmodels

        return [
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
