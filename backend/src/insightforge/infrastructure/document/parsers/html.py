"""HTML document parser."""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from insightforge.core.logging import get_logger
from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document.base import DocumentParser
from insightforge.infrastructure.document.helpers import ensure_text, merge_metadata
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)


class HtmlDocumentParser(DocumentParser):
    """Extract readable text and light metadata from HTML."""

    name = ContentType.HTML

    def parse(
        self,
        raw: bytes | str,
        *,
        source_url: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        html = ensure_text(raw)
        logger.info("html parse started chars=%d", len(html))

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()

        page_title = title
        if not page_title and soup.title and soup.title.string:
            page_title = soup.title.string.strip() or None

        description: str | None = None
        for meta in soup.find_all("meta"):
            name = str(meta.get("name") or "").lower()
            if name == "description":
                content = meta.get("content")
                if content:
                    description = str(content).strip() or None
                break

        # Prefer article/main; strip common chrome before text extraction.
        root = soup.find("article") or soup.find("main") or soup.body or soup
        for tag in root.find_all(["nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = root.get_text(separator="\n", strip=True)

        metadata = merge_metadata(
            parser=self.name.value,
            filename=filename,
            extra=extra_metadata,
            extracted={
                "description": description,
                "char_count": len(text),
            },
        )
        logger.info(
            "html parse finished chars=%d title=%s",
            len(text),
            bool(page_title),
        )
        return ParsedDocument(
            text=text,
            content_type=self.name,
            title=page_title,
            url=source_url,
            metadata=metadata,
        )
