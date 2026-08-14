"""JSON document parser."""

from __future__ import annotations

import json
from typing import Any

from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document.base import DocumentParser
from insightforge.infrastructure.document.helpers import ensure_text, merge_metadata
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)

_CONTENT_KEYS = ("content", "text", "body", "markdown", "html", "article")
_TITLE_KEYS = ("title", "name", "headline")
_AUTHOR_KEYS = ("author", "authors", "by")
_DATE_KEYS = ("date", "published", "published_at", "created", "created_at")


def _first_string(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _flatten(value: Any, *, prefix: str = "") -> list[str]:
    """Turn nested JSON into readable ``key: value`` lines."""

    lines: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten(child, prefix=path))
        return lines
    if isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            lines.extend(_flatten(child, prefix=path))
        return lines
    if value is None:
        return lines
    label = prefix or "value"
    lines.append(f"{label}: {value}")
    return lines


def _json_author_date(data: Any) -> tuple[str | None, str | None]:
    if not isinstance(data, dict):
        return None, None
    author = _first_string(data, _AUTHOR_KEYS)
    if author is None:
        raw_authors = data.get("authors")
        if isinstance(raw_authors, list):
            names = [str(item).strip() for item in raw_authors if str(item).strip()]
            author = "; ".join(names) or None
    published = _first_string(data, _DATE_KEYS)
    return author, published


def _json_to_text(data: Any) -> tuple[str, str | None]:
    if isinstance(data, str):
        return data.strip(), None
    if isinstance(data, dict):
        title = _first_string(data, _TITLE_KEYS)
        content = _first_string(data, _CONTENT_KEYS)
        if content is not None:
            return content, title
        return "\n".join(_flatten(data)), title
    if isinstance(data, list):
        return "\n".join(_flatten(data)), None
    return str(data), None


class JsonDocumentParser(DocumentParser):
    """Extract text from JSON payloads (content fields or flattened values)."""

    name = ContentType.JSON

    def parse(
        self,
        raw: bytes | str,
        *,
        source_url: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        source = ensure_text(raw)
        logger.info("json parse started chars=%d", len(source))

        try:
            data = json.loads(source)
        except json.JSONDecodeError as exc:
            logger.warning("json parse failed error=%s", exc)
            raise ValidationFailedError(
                "invalid JSON payload",
                details={"parser": self.name.value, "error": str(exc)},
            ) from exc

        text, inferred_title = _json_to_text(data)
        page_title = title or inferred_title
        author, published = _json_author_date(data)
        metadata = merge_metadata(
            parser=self.name.value,
            filename=filename,
            extra=extra_metadata,
            extracted={
                "root_type": type(data).__name__,
                "author": author,
                "date": published,
                "char_count": len(text),
            },
        )
        logger.info(
            "json parse finished chars=%d title=%s root=%s",
            len(text),
            bool(page_title),
            type(data).__name__,
        )
        return ParsedDocument(
            text=text,
            content_type=self.name,
            title=page_title,
            url=source_url,
            metadata=metadata,
        )
