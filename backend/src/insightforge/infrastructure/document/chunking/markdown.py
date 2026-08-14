"""Markdown-aware chunker — split on headings, then recurse oversized sections."""

from __future__ import annotations

import re
from dataclasses import dataclass

from insightforge.core.logging import get_logger
from insightforge.domain.models import DocumentChunk, ParsedDocument
from insightforge.infrastructure.document.chunking.base import DocumentChunker
from insightforge.infrastructure.document.chunking.helpers import build_chunks
from insightforge.infrastructure.document.chunking.recursive import split_recursive
from insightforge.shared.enums import ChunkStrategy

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class _Section:
    heading: str | None
    path: tuple[str, ...]
    text: str


class MarkdownAwareChunker(DocumentChunker):
    """Keep markdown heading sections together; recurse only when they overflow."""

    name = ChunkStrategy.MARKDOWN

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.text
        if not text.strip():
            logger.info("markdown chunker skipped empty document")
            return []

        sections = _iter_sections(text)
        pieces: list[tuple[str, str | None]] = []
        for section in sections:
            heading_label = " > ".join(section.path) if section.path else None
            if len(section.text) <= self.config.chunk_size:
                pieces.append((section.text, heading_label))
                continue
            splits = split_recursive(
                section.text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            for split in splits:
                pieces.append((split, heading_label))

        chunks = build_chunks(
            document,
            pieces,
            strategy=self.name,
            extra_metadata={
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            },
        )
        logger.info(
            "markdown chunker finished chars=%d sections=%d chunks=%d",
            len(text),
            len(sections),
            len(chunks),
        )
        return chunks


def _iter_sections(text: str) -> list[_Section]:
    """Split markdown into heading sections, ignoring headings inside fences."""

    lines = text.splitlines(keepends=True)
    in_fence = False
    stack: list[tuple[int, str]] = []
    current_parts: list[str] = []
    current_heading: str | None = None
    current_path: tuple[str, ...] = ()
    sections: list[_Section] = []

    def flush() -> None:
        body = "".join(current_parts)
        if body.strip():
            sections.append(_Section(heading=current_heading, path=current_path, text=body.strip()))

    for line in lines:
        stripped = line.strip("\n")
        fence_marker = stripped.lstrip()
        if fence_marker.startswith("```") or fence_marker.startswith("~~~"):
            in_fence = not in_fence
            current_parts.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(stripped)
        if match is None:
            current_parts.append(line)
            continue

        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current_heading = title
        current_path = tuple(item[1] for item in stack)
        current_parts = [line]

    flush()
    return sections
