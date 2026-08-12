"""Heuristic relevance scoring for retrieved documents.

``Document.score`` is the application relevance score (query match + content
richness). Provider-native signals such as Tavily ``metadata["score"]`` or
GitHub ``metadata["stars"]`` stay in metadata and are not mixed into
``Document.score``. Provider identity is never used as a relevance prior.
"""

from __future__ import annotations

import re

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _keyword_overlap(query: str, document: Document) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    haystack = _tokens(f"{document.title} {document.snippet} {document.content}")
    if not haystack:
        return 0.0
    return len(query_tokens & haystack) / len(query_tokens)


def _content_richness(document: Document) -> float:
    """Normalize available text length into 0..1 (not a provider prior)."""

    return min(1.0, (len(document.snippet) + len(document.content)) / 800.0)


def provider_metadata_score(document: Document) -> float | None:
    """Return a normalized 0..1 view of provider-native metadata, if present.

    This is intentionally separate from ``Document.score`` so callers can
    inspect provider signals without treating them as application relevance.
    """

    meta = document.metadata
    if not meta:
        return None

    if isinstance(meta.get("score"), int | float):
        raw = float(meta["score"])
        # Tavily scores are often 0..1; Reddit upvote totals can be larger.
        if 0.0 <= raw <= 1.0:
            return raw
        return min(1.0, raw / 100.0)

    if isinstance(meta.get("provider_score"), int | float):
        raw = float(meta["provider_score"])
        return min(1.0, max(0.0, raw if raw <= 1.0 else raw / 100.0))

    if isinstance(meta.get("stars"), int | float):
        return min(1.0, float(meta["stars"]) / 5000.0)

    return None


def score_document(document: Document, query: str) -> float:
    """Return a 0..1 application relevance score for ``document`` vs ``query``.

    Uses only query/content signals. Provider identity and provider metadata
    scores do not contribute.
    """

    keyword = _keyword_overlap(query, document)
    richness = _content_richness(document)
    score = (0.80 * keyword) + (0.20 * richness)
    return round(min(1.0, max(0.0, score)), 4)


def score_documents(documents: list[Document], query: str) -> list[Document]:
    """Attach application relevance scores and return documents best-first."""

    scored: list[Document] = []
    for doc in documents:
        scored.append(doc.model_copy(update={"score": score_document(doc, query)}))

    scored.sort(key=lambda item: item.score or 0.0, reverse=True)
    if scored:
        logger.debug(
            "scored documents count=%d top_score=%.4f",
            len(scored),
            scored[0].score or 0.0,
        )
    return scored
