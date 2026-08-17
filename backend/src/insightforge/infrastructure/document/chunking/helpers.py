"""Shared helpers for document chunkers."""

from __future__ import annotations

import re
from typing import Any

from insightforge.domain.models import DocumentChunk, ParsedDocument
from insightforge.shared.enums import ChunkStrategy

_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
    r"|\n{2,}"
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*", re.IGNORECASE)
_INHERIT_KEYS = ("parser", "filename", "author", "date", "provider")


def locate_span(source: str, piece: str, *, start_at: int = 0) -> tuple[int, int]:
    """Return ``(start, end)`` of ``piece`` inside ``source``."""

    if not piece:
        return start_at, start_at
    index = source.find(piece, max(0, start_at))
    if index < 0:
        index = source.find(piece)
    if index < 0:
        return start_at, start_at + len(piece)
    return index, index + len(piece)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(sentence, start, end)`` spans."""

    if not text.strip():
        return []

    spans: list[tuple[str, int, int]] = []
    last = 0
    for match in _SENTENCE_RE.finditer(text):
        _append_span(spans, text, last, match.start())
        last = match.end()
    _append_span(spans, text, last, len(text))
    return spans


def tokenize(text: str) -> frozenset[str]:
    """Lowercased word-like tokens for lexical similarity."""

    return frozenset(token.lower() for token in _TOKEN_RE.findall(text))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def inherited_metadata(
    document: ParsedDocument,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a small allowlist of source metadata onto each chunk."""

    metadata: dict[str, Any] = {}
    for key in _INHERIT_KEYS:
        value = document.metadata.get(key)
        if value is not None and value != "":
            metadata[key] = value
    if extra:
        metadata.update(extra)
    return metadata


def build_chunks(
    document: ParsedDocument,
    pieces: list[tuple[str, str | None]],
    *,
    strategy: ChunkStrategy,
    extra_metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:
    """Turn ``(text, heading)`` pieces into indexed ``DocumentChunk`` rows."""

    located: list[tuple[str, int, int, str | None]] = []
    # Advance past each match start so repeated piece text binds to later
    # occurrences, while overlap windows (which begin inside the prior span)
    # still resolve correctly when searching from ``start + 1``.
    cursor = 0
    for piece, heading in pieces:
        text = piece.strip()
        if not text:
            continue
        start, end = locate_span(document.text, text, start_at=cursor)
        located.append((text, start, end, heading))
        cursor = start + 1

    inherited = inherited_metadata(document, extra_metadata)
    total = len(located)
    chunks: list[DocumentChunk] = []
    for index, (text, start, end, heading) in enumerate(located):
        metadata = dict(inherited)
        metadata["total_chunks"] = total
        metadata["char_count"] = len(text)
        chunks.append(
            DocumentChunk(
                text=text,
                index=index,
                start=start,
                end=end,
                strategy=strategy,
                title=document.title,
                url=document.url,
                heading=heading,
                content_type=document.content_type,
                metadata=metadata,
            )
        )
    return chunks


def _append_span(
    spans: list[tuple[str, int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    region = text[start:end]
    piece = region.strip()
    if not piece:
        return
    rel = region.find(piece)
    abs_start = start + rel
    spans.append((piece, abs_start, abs_start + len(piece)))
