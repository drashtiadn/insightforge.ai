"""Unit tests for vector stores, sessions, and VectorStoreService."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient

from insightforge.core.config import Settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.domain.models import VectorRecord
from insightforge.infrastructure.vectorstores import (
    MemoryVectorStore,
    QdrantVectorStore,
    SessionVectorStoreRegistry,
    VectorStoreService,
    cosine_similarity,
    create_vector_store_service,
    parse_vector_store,
    require_records,
    require_vector,
)
from insightforge.infrastructure.vectorstores.helpers import require_dimensions
from insightforge.infrastructure.vectorstores.stores.qdrant import point_id_for
from insightforge.shared.enums import VectorStoreHint


def _record(
    record_id: str,
    vector: list[float],
    *,
    text: str = "",
    **metadata: Any,
) -> VectorRecord:
    return VectorRecord(id=record_id, vector=vector, text=text, metadata=dict(metadata))


def test_require_vector_rejects_empty() -> None:
    with pytest.raises(ValidationFailedError):
        require_vector([])


def test_require_records_rejects_blank_id() -> None:
    with pytest.raises(ValidationFailedError, match="id"):
        require_records([_record("  ", [1.0, 0.0])])


def test_require_dimensions_rejects_mixed_batch() -> None:
    with pytest.raises(ValidationFailedError, match="inconsistent"):
        require_dimensions([[1.0, 0.0], [1.0]], expected=None)


def test_parse_vector_store() -> None:
    assert parse_vector_store("memory") is VectorStoreHint.MEMORY
    assert parse_vector_store("QDRANT") is VectorStoreHint.QDRANT
    with pytest.raises(ValidationFailedError):
        parse_vector_store("faiss")


def test_cosine_similarity_edges() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    with pytest.raises(ValidationFailedError):
        cosine_similarity([1.0], [1.0, 0.0])


def test_memory_upsert_search_filter_delete() -> None:
    store = MemoryVectorStore(dimensions=2)
    written = store.upsert(
        [
            _record("a", [1.0, 0.0], text="alpha", source="web"),
            _record("b", [0.0, 1.0], text="beta", source="arxiv"),
            _record("c", [0.9, 0.1], text="gamma", source="web"),
        ]
    )
    assert written == 3
    assert store.count() == 3
    assert store.count(filters={"source": "web"}) == 2

    hits = store.search([1.0, 0.0], limit=2)
    assert [hit.id for hit in hits] == ["a", "c"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].text == "alpha"

    filtered = store.search([1.0, 0.0], limit=5, filters={"source": "arxiv"})
    assert len(filtered) == 1
    assert filtered[0].id == "b"

    removed = store.delete(ids=["b"])
    assert removed == 1
    assert store.count() == 2

    removed = store.delete(filters={"source": "web"})
    assert removed == 2
    assert store.count() == 0


def test_memory_upsert_overwrites_same_id() -> None:
    store = MemoryVectorStore(dimensions=2)
    store.upsert([_record("a", [1.0, 0.0], text="first")])
    store.upsert([_record("a", [0.0, 1.0], text="second")])
    assert store.count() == 1
    hit = store.search([0.0, 1.0], limit=1)[0]
    assert hit.text == "second"


def test_memory_empty_upsert_and_noop_delete() -> None:
    store = MemoryVectorStore(dimensions=2)
    assert store.upsert([]) == 0
    assert store.delete() == 0
    assert store.search([1.0, 0.0], limit=0) == []


def test_memory_rejects_dimension_mismatch() -> None:
    store = MemoryVectorStore(dimensions=2)
    store.upsert([_record("a", [1.0, 0.0])])
    with pytest.raises(ValidationFailedError):
        store.upsert([_record("b", [1.0, 0.0, 0.0])])
    with pytest.raises(ValidationFailedError):
        store.search([1.0])


def test_memory_score_threshold_and_with_vectors() -> None:
    store = MemoryVectorStore()
    store.upsert(
        [
            _record("a", [1.0, 0.0]),
            _record("b", [0.0, 1.0]),
        ]
    )
    hits = store.search([1.0, 0.0], limit=5, score_threshold=0.5, with_vectors=True)
    assert len(hits) == 1
    assert hits[0].id == "a"
    assert hits[0].vector == [1.0, 0.0]


def test_memory_clear() -> None:
    store = MemoryVectorStore(dimensions=2)
    store.upsert([_record("a", [1.0, 0.0])])
    store.clear()
    assert store.count() == 0


def test_session_registry_isolates_and_expires() -> None:
    registry = SessionVectorStoreRegistry(ttl_seconds=60.0, dimensions=2)
    left = registry.get("session-a")
    right = registry.get("session-b")
    left.upsert([_record("1", [1.0, 0.0], text="only-a")])
    right.upsert([_record("1", [0.0, 1.0], text="only-b")])

    assert left.count() == 1
    assert right.count() == 1
    assert left.search([1.0, 0.0], limit=1)[0].text == "only-a"
    assert right.search([0.0, 1.0], limit=1)[0].text == "only-b"
    assert registry.get("session-a") is left

    assert registry.drop("session-a") is True
    assert registry.drop("missing") is False
    assert len(registry) == 1

    entry = registry._sessions["session-b"]
    entry.last_access = entry.last_access - 120.0
    assert registry.cleanup() == 1
    assert len(registry) == 0


def test_session_get_drops_expired_without_cleanup() -> None:
    registry = SessionVectorStoreRegistry(ttl_seconds=10.0, dimensions=2)
    store = registry.get("s1")
    store.upsert([_record("1", [1.0, 0.0])])
    entry = registry._sessions["s1"]
    entry.last_access = entry.last_access - 30.0
    fresh = registry.get("s1")
    assert fresh is not store
    assert fresh.count() == 0


def test_session_registry_rejects_blank_id() -> None:
    registry = SessionVectorStoreRegistry(ttl_seconds=10.0)
    with pytest.raises(ValidationFailedError):
        registry.get("  ")
    with pytest.raises(ValidationFailedError):
        SessionVectorStoreRegistry(ttl_seconds=0)


def test_session_registry_clear() -> None:
    registry = SessionVectorStoreRegistry(ttl_seconds=60.0, dimensions=2)
    registry.get("a").upsert([_record("1", [1.0, 0.0])])
    registry.clear()
    assert len(registry) == 0


def test_point_id_for_uuid_and_hash() -> None:
    uuid = "11111111-1111-1111-1111-111111111111"
    assert point_id_for(uuid) == uuid
    assert point_id_for("chunk-1") != "chunk-1"
    assert point_id_for("chunk-1") == point_id_for("chunk-1")


def test_qdrant_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValidationFailedError):
        QdrantVectorStore(client=QdrantClient(location=":memory:"), dimensions=0)


def test_qdrant_in_memory_upsert_search_filter_delete() -> None:
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=client, collection="unit", dimensions=2)

    written = store.upsert(
        [
            _record("doc-1", [1.0, 0.0], text="hello", source="web"),
            _record("doc-2", [0.0, 1.0], text="world", source="arxiv"),
            _record("doc-3", [0.95, 0.05], text="near", source="web"),
        ]
    )
    assert written == 3
    assert store.count() == 3
    assert store.count(filters={"source": "web"}) == 2

    hits = store.search([1.0, 0.0], limit=2)
    assert hits[0].id == "doc-1"
    assert hits[0].text == "hello"
    assert hits[0].metadata["source"] == "web"
    assert hits[0].score > hits[1].score

    filtered = store.search([1.0, 0.0], limit=5, filters={"source": "arxiv"})
    assert len(filtered) == 1
    assert filtered[0].id == "doc-2"

    thresholded = store.search([1.0, 0.0], limit=5, score_threshold=0.5)
    ids = [hit.id for hit in thresholded]
    assert "doc-1" in ids
    assert "doc-2" not in ids

    with_vecs = store.search([1.0, 0.0], limit=1, with_vectors=True)
    assert with_vecs[0].vector is not None
    assert len(with_vecs[0].vector) == 2

    removed = store.delete(ids=["doc-2"])
    assert removed == 1
    assert store.count() == 2

    removed = store.delete(filters={"source": "web"})
    assert removed == 2
    assert store.count() == 0

    store.upsert([_record("keep", [1.0, 0.0], text="x")])
    store.clear()
    assert store.count() == 0


def test_qdrant_upsert_overwrites_same_id() -> None:
    store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection="overwrite",
        dimensions=2,
    )
    store.upsert([_record("same", [1.0, 0.0], text="first")])
    store.upsert([_record("same", [0.0, 1.0], text="second")])
    assert store.count() == 1
    assert store.search([0.0, 1.0], limit=1)[0].text == "second"


def test_qdrant_maps_client_errors() -> None:
    class _Broken:
        def collection_exists(self, name: str) -> bool:
            return True

        def upsert(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

        def query_points(self, **kwargs: Any) -> SimpleNamespace:
            raise RuntimeError("boom")

    store = QdrantVectorStore(client=_Broken(), collection="broken", dimensions=2)
    with pytest.raises(ExternalServiceError, match="upsert"):
        store.upsert([_record("a", [1.0, 0.0])])
    with pytest.raises(ExternalServiceError, match="search"):
        store.search([1.0, 0.0], limit=1)


def test_create_vector_store_service_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("VECTOR_DIMENSIONS", "2")
    settings = Settings(_env_file=None)

    service = create_vector_store_service(settings)
    assert service.store.name is VectorStoreHint.MEMORY
    assert service.available
    service.upsert([_record("x", [1.0, 0.0], text="session-ready")])
    assert service.search([1.0, 0.0], limit=1)[0].id == "x"

    session = service.session("research-1")
    session.upsert([_record("y", [0.0, 1.0], text="temp")])
    assert service.count() == 1
    assert session.count() == 1
    assert service.drop_session("research-1") is True
    assert service.cleanup_sessions() == 0


def test_create_vector_store_service_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("VECTOR_DIMENSIONS", "2")
    monkeypatch.setenv("QDRANT_COLLECTION", "unit")
    settings = Settings(_env_file=None)

    client = QdrantClient(location=":memory:")
    service = create_vector_store_service(settings, qdrant_client=client)
    assert service.store.name is VectorStoreHint.QDRANT
    service.upsert([_record("z", [1.0, 0.0], text="q")])
    assert service.search([1.0, 0.0], limit=1)[0].text == "q"


def test_service_raises_when_unavailable() -> None:
    class _DownStore(MemoryVectorStore):
        @property
        def available(self) -> bool:
            return False

    service = VectorStoreService(_DownStore(dimensions=2))
    with pytest.raises(ExternalServiceError, match="unavailable"):
        service.upsert([_record("a", [1.0, 0.0])])


def test_vector_store_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.vector_store == "memory"
    assert settings.vector_dimensions == 1024
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "insightforge"
    assert settings.vector_session_ttl_seconds == 3600.0
