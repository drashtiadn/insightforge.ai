"""Light metadata extraction from cleaned document text (Phase 4.2).

Citation structuring (canonical author/date/title objects) is Phase 4.4.
This step only fills missing metadata fields discovered in the text body.
"""

from __future__ import annotations

import re
from typing import Any

from insightforge.core.logging import get_logger

logger = get_logger(__name__)

_URL_RE = re.compile(
    r"https?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)
_AUTHOR_RE = re.compile(
    r"(?im)^\s*(?:by|author|written\s+by)\s*[:\-]?\s+([A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+){0,3})\s*$"
)
_DATE_RE = re.compile(
    r"(?ix)(?:^|\b)("
    r"\d{4}-\d{2}-\d{2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    r"\s+\d{4}"
    r")\b"
)
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def extract_text_metadata(
    text: str,
    *,
    existing: dict[str, Any] | None = None,
    title: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Return metadata enriched with fields discovered in ``text``.

    Existing non-empty values win. Counts always reflect the cleaned text.
    """

    metadata = dict(existing or {})
    found: list[str] = []

    if not _has_value(metadata.get("author")):
        author = _first_match(_AUTHOR_RE, text)
        if author:
            metadata["author"] = author
            found.append("author")

    if not _has_value(metadata.get("date")) and not _has_value(metadata.get("published")):
        date = _first_match(_DATE_RE, text)
        if date:
            metadata["date"] = date
            found.append("date")

    urls = _unique(_URL_RE.findall(text))
    if url:
        urls = [item for item in urls if item.rstrip("/") != url.rstrip("/")]
    if urls and not _has_value(metadata.get("urls_in_text")):
        metadata["urls_in_text"] = urls[:20]
        found.append("urls_in_text")

    emails = _unique(_EMAIL_RE.findall(text))
    if emails and not _has_value(metadata.get("emails")):
        metadata["emails"] = emails[:10]
        found.append("emails")

    words = _WORD_RE.findall(text)
    metadata["char_count"] = len(text)
    metadata["word_count"] = len(words)
    metadata["cleaned"] = True

    if title and not _has_value(metadata.get("title")):
        metadata["title"] = title

    logger.debug(
        "extract_text_metadata found=%s words=%d chars=%d",
        found,
        metadata["word_count"],
        metadata["char_count"],
    )
    return metadata


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | dict | tuple | set):
        return len(value) > 0
    return True


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip() or None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.rstrip(".,);]")
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result
