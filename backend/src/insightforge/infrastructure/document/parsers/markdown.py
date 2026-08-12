"""Markdown document parser."""

from __future__ import annotations

import re
from typing import Any

from insightforge.core.logging import get_logger
from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document.base import DocumentParser
from insightforge.infrastructure.document.helpers import ensure_text, merge_metadata
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z",
    re.DOTALL,
)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FRONTMATTER_LINE_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")


def _parse_frontmatter(block: str) -> dict[str, str]:
    """Parse simple ``key: value`` YAML-like frontmatter without a YAML dependency."""

    fields: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _FRONTMATTER_LINE_RE.match(stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        fields[key.lower()] = value
    return fields


class MarkdownDocumentParser(DocumentParser):
    """Extract body text and light metadata from Markdown sources."""

    name = ContentType.MARKDOWN

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
        logger.info("markdown parse started chars=%d", len(source))

        frontmatter: dict[str, str] = {}
        body = source
        match = _FRONTMATTER_RE.match(source)
        if match:
            frontmatter = _parse_frontmatter(match.group(1))
            body = match.group(2)

        page_title = title or frontmatter.get("title")
        if not page_title:
            heading = _HEADING_RE.search(body)
            if heading:
                page_title = heading.group(1).strip() or None

        text = body.strip()
        metadata = merge_metadata(
            parser=self.name.value,
            filename=filename,
            extra=extra_metadata,
            extracted={
                "frontmatter": frontmatter or None,
                "author": frontmatter.get("author"),
                "date": frontmatter.get("date") or frontmatter.get("published"),
                "char_count": len(text),
            },
        )
        logger.info(
            "markdown parse finished chars=%d title=%s frontmatter_keys=%d",
            len(text),
            bool(page_title),
            len(frontmatter),
        )
        return ParsedDocument(
            text=text,
            content_type=self.name,
            title=page_title,
            url=source_url,
            metadata=metadata,
        )
