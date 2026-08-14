"""Shared helpers for document parsers."""

from __future__ import annotations

from typing import Any

from insightforge.core.exceptions import ValidationFailedError


def ensure_bytes(raw: bytes | str, *, encoding: str = "utf-8") -> bytes:
    """Normalize parser input to bytes."""

    if isinstance(raw, bytes):
        if not raw:
            raise ValidationFailedError(
                "document payload must not be empty",
                details={"field": "raw"},
            )
        return raw
    text = raw.strip("\ufeff")
    if not text:
        raise ValidationFailedError(
            "document payload must not be empty",
            details={"field": "raw"},
        )
    return text.encode(encoding)


def ensure_text(raw: bytes | str, *, encoding: str = "utf-8") -> str:
    """Normalize parser input to Unicode text."""

    if isinstance(raw, str):
        text = raw.strip("\ufeff")
        if not text.strip():
            raise ValidationFailedError(
                "document payload must not be empty",
                details={"field": "raw"},
            )
        return text
    if not raw:
        raise ValidationFailedError(
            "document payload must not be empty",
            details={"field": "raw"},
        )
    return raw.decode(encoding, errors="replace").strip("\ufeff")


def merge_metadata(
    *,
    parser: str,
    filename: str | None,
    extra: dict[str, Any] | None,
    extracted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata with parser identity and optional caller fields."""

    metadata: dict[str, Any] = {"parser": parser}
    if filename:
        metadata["filename"] = filename
    if extracted:
        metadata.update({key: value for key, value in extracted.items() if value is not None})
    if extra:
        metadata.update(extra)
    return metadata
