"""Heuristic relevance scoring for retrieved documents."""

from __future__ import annotations

import re

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*", re.IGNORECASE)

PROVIDER_WEIGHTS: dict[SearchProviderHint, float] = {
    SearchProviderHint.ARXIV: 0.92,
    SearchProviderHint.WIKIPEDIA: 0.88,
    SearchProviderHint.WEB: 0.72,
    SearchProviderHint.GITHUB: 0.78,
    SearchProviderHint.YOUTUBE: 0.62,
    SearchProviderHint.REDDIT: 0.55,
}


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


def _metadata_signal(document: Document) -> float:
    """Normalize optional provider-specific signals into 0..1."""

    meta = document.metadata
    if not meta:
        return 0.0

    if isinstance(meta.get("score"), int | float):
        raw = float(meta["score"])
        # Tavily scores are often 0..1; Reddit can be larger.
        if 0.0 <= raw <= 1.0:
            return raw
        return min(1.0, raw / 100.0)

    if isinstance(meta.get("stars"), int | float):
        stars = float(meta["stars"])
        return min(1.0, stars / 5000.0)

    return 0.0


def score_document(document: Document, query: str) -> float:
    """Return a 0..1 relevance score for ``document`` against ``query``."""

    keyword = _keyword_overlap(query, document)
    provider = PROVIDER_WEIGHTS.get(document.provider, 0.5)
    meta = _metadata_signal(document)
    richness = min(1.0, (len(document.snippet) + len(document.content)) / 800.0)

    score = (0.40 * keyword) + (0.30 * provider) + (0.20 * meta) + (0.10 * richness)
    return round(min(1.0, max(0.0, score)), 4)


def score_documents(documents: list[Document], query: str) -> list[Document]:
    """Attach scores and return documents sorted best-first."""

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
