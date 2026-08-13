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

_AUTHOR_META = frozenset(
    {
        "author",
        "citation_author",
        "article:author",
        "dc.creator",
        "dcterms.creator",
        "og:article:author",
    }
)
_DATE_META = frozenset(
    {
        "date",
        "pubdate",
        "publish_date",
        "publication_date",
        "citation_publication_date",
        "article:published_time",
        "og:article:published_time",
        "dc.date",
        "dcterms.issued",
        "parsely-pub-date",
    }
)


def _meta_key(tag: Any) -> str:
    return str(tag.get("name") or tag.get("property") or tag.get("itemprop") or "").lower()


def _extract_html_meta(soup: BeautifulSoup) -> tuple[str | None, str | None, str | None]:
    """Return ``(description, author, date)`` from HTML meta tags."""

    description: str | None = None
    authors: list[str] = []
    published: str | None = None
    for meta in soup.find_all("meta"):
        key = _meta_key(meta)
        content = meta.get("content")
        if not content:
            continue
        value = str(content).strip()
        if not value:
            continue
        if description is None and key in {"description", "og:description"}:
            description = value
        if key in _AUTHOR_META or key.endswith(":author"):
            authors.append(value)
        elif published is None and (key in _DATE_META or "published" in key):
            published = value
    author = "; ".join(dict.fromkeys(authors)) or None
    return description, author, published


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

        description, author, published = _extract_html_meta(soup)

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
                "author": author,
                "date": published,
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
