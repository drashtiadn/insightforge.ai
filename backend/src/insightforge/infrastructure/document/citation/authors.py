"""Split free-text author fields into a list of names."""

from __future__ import annotations

import re
from typing import Any

_SPLIT_RE = re.compile(r"\s*(?:;|&|\band\b)\s*", re.IGNORECASE)


def parse_authors(value: Any) -> list[str]:
    """Normalize ``author`` / ``authors`` metadata into distinct names."""

    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        names: list[str] = []
        for item in value:
            names.extend(parse_authors(item))
        return _dedupe(names)

    text = str(value).strip()
    if not text:
        return []

    parts = [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]
    if len(parts) == 1:
        parts = _split_comma_list(parts[0])
    return _dedupe(parts)


def _split_comma_list(text: str) -> list[str]:
    """Split ``A, B, C`` but keep ``Last, First`` as one name."""

    if "," not in text:
        return [text]
    if text.count(",") == 1:
        left, right = (part.strip() for part in text.split(",", 1))
        if left and right and " " not in left and len(right.split()) <= 3:
            return [f"{left}, {right}"]
    return [part.strip() for part in text.split(",") if part.strip()]


def _dedupe(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        key = re.sub(r"\s+", " ", name).strip()
        if not key:
            continue
        marker = key.lower()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(key)
    return result
