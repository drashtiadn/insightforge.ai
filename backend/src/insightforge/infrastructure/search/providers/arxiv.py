"""ArXiv paper search via the Atom API."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.infrastructure.search.base import SearchProvider
from insightforge.infrastructure.search.http import raise_for_status, require_query
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivSearchProvider(SearchProvider):
    """Search arXiv preprints."""

    name = SearchProviderHint.ARXIV

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        cleaned = require_query(query)
        logger.info("arxiv search query_len=%d limit=%d", len(cleaned), limit)

        response = self._client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{cleaned}",
                "start": 0,
                "max_results": limit,
            },
        )
        raise_for_status(response, provider="arxiv")

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise ExternalServiceError(
                "arxiv returned invalid Atom XML",
                details={"provider": "arxiv"},
            ) from exc

        documents: list[Document] = []
        for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
            title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
            title = " ".join(title.split())
            summary = (
                entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or ""
            ).strip()
            summary = " ".join(summary.split())
            url = ""
            for link in entry.findall("atom:link", _ATOM_NS):
                href = link.attrib.get("href", "")
                rel = link.attrib.get("rel", "")
                link_type = link.attrib.get("type", "")
                if rel == "alternate" or link_type == "text/html":
                    url = href
                    break
                if not url and href:
                    url = href
            if not title or not url:
                continue
            published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
            documents.append(
                Document(
                    title=title,
                    url=url,
                    snippet=summary[:400],
                    content=summary,
                    provider=self.name,
                    metadata={"published": published.strip()},
                )
            )

        logger.info("arxiv returned count=%d", len(documents))
        return documents
