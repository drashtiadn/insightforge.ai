"""Shared helpers for vector stores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any

from insightforge.core.exceptions import ValidationFailedError
from insightforge.domain.models import VectorRecord


def require_vector(vector: Sequence[float], *, field: str = "vector") -> list[float]:
    """Normalize a query/document vector and reject empties."""

    values = [float(value) for value in vector]
    if not values:
        raise ValidationFailedError(
            "vector must not be empty",
            details={"field": field},
        )
    return values


def require_records(records: Sequence[VectorRecord]) -> list[VectorRecord]:
    """Validate upsert payloads; empty input is allowed."""

    cleaned: list[VectorRecord] = []
    for index, record in enumerate(records):
        if not record.id.strip():
            raise ValidationFailedError(
                "vector record id must not be empty",
                details={"field": "id", "index": index},
            )
        if not record.vector:
            raise ValidationFailedError(
                "vector record vector must not be empty",
                details={"field": "vector", "index": index, "id": record.id},
            )
        cleaned.append(record)
    return cleaned


def require_dimensions(vectors: Sequence[Sequence[float]], *, expected: int | None) -> int:
    """Ensure all vectors share one length; optionally match ``expected``."""

    if not vectors:
        if expected is None:
            raise ValidationFailedError(
                "cannot infer vector dimensions from empty input",
                details={"field": "dimensions"},
            )
        return expected

    dims = len(vectors[0])
    for index, vector in enumerate(vectors):
        if len(vector) != dims:
            raise ValidationFailedError(
                "inconsistent vector dimensions in batch",
                details={
                    "field": "vector",
                    "index": index,
                    "expected": dims,
                    "actual": len(vector),
                },
            )
    if expected is not None and dims != expected:
        raise ValidationFailedError(
            "vector dimensions do not match store configuration",
            details={"field": "vector", "expected": expected, "actual": dims},
        )
    return dims


def matches_filters(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """Equality filter: every ``filters`` key must match ``metadata``."""

    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``. Returns ``0.0`` for zero vectors."""

    if len(left) != len(right):
        raise ValidationFailedError(
            "cosine similarity requires equal-length vectors",
            details={"left": len(left), "right": len(right)},
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (sqrt(left_norm) * sqrt(right_norm))
