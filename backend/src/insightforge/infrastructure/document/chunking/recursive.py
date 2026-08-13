"""Recursive character chunker."""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import DocumentChunk, ParsedDocument
from insightforge.infrastructure.document.chunking.base import ChunkConfig, DocumentChunker
from insightforge.infrastructure.document.chunking.helpers import build_chunks
from insightforge.shared.enums import ChunkStrategy

logger = get_logger(__name__)

DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
)


class RecursiveCharacterChunker(DocumentChunker):
    """Split on the coarsest separator that keeps pieces under ``chunk_size``."""

    name = ChunkStrategy.RECURSIVE

    def __init__(
        self,
        config: ChunkConfig | None = None,
        *,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        super().__init__(config)
        self._separators = tuple(separators) or DEFAULT_SEPARATORS

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.text
        if not text.strip():
            logger.info("recursive chunker skipped empty document")
            return []

        pieces = split_recursive(
            text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self._separators,
        )
        chunks = build_chunks(
            document,
            [(piece, None) for piece in pieces],
            strategy=self.name,
            extra_metadata={
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            },
        )
        logger.info(
            "recursive chunker finished chars=%d chunks=%d size=%d overlap=%d",
            len(text),
            len(chunks),
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        return chunks


def split_recursive(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Recursively split ``text`` into pieces of at most ``chunk_size`` characters."""

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    return _split(cleaned, chunk_size, chunk_overlap, list(separators))


def _split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    separator, rest = _choose_separator(text, separators)
    raw_parts = list(text) if separator == "" else text.split(separator)

    good: list[str] = []
    pending: list[str] = []
    for part in raw_parts:
        if len(part) > chunk_size:
            if pending:
                good.extend(_merge(pending, separator, chunk_size, chunk_overlap))
                pending = []
            if rest:
                good.extend(_split(part, chunk_size, chunk_overlap, rest))
            else:
                good.extend(_hard_wrap(part, chunk_size, chunk_overlap))
        else:
            pending.append(part)
    if pending:
        good.extend(_merge(pending, separator, chunk_size, chunk_overlap))
    return [piece for piece in good if piece.strip()]


def _choose_separator(text: str, separators: list[str]) -> tuple[str, list[str]]:
    if not separators:
        return "", []
    for index, separator in enumerate(separators):
        if separator == "":
            return "", []
        if separator in text:
            return separator, separators[index + 1 :]
    return separators[-1], []


def _merge(
    pieces: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not pieces:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for piece in pieces:
        candidate = [*current, piece]
        length = len(separator.join(candidate))
        if current and length > chunk_size:
            chunks.append(separator.join(current))
            current = _overlap_window(current, separator, chunk_overlap)
            candidate = [*current, piece]
            if current and len(separator.join(candidate)) > chunk_size:
                current = []
                candidate = [piece]
        current = candidate

    if current:
        joined = separator.join(current)
        if joined.strip():
            chunks.append(joined)
    return chunks


def _overlap_window(pieces: list[str], separator: str, overlap: int) -> list[str]:
    if overlap <= 0 or not pieces:
        return []
    window: list[str] = []
    for piece in reversed(pieces):
        candidate = [piece, *window]
        length = len(separator.join(candidate))
        if window and length > overlap:
            break
        window = candidate
    return window


def _hard_wrap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks
