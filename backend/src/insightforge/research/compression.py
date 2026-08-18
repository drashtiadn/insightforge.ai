"""Context compression — trim ranked evidence down to a reasoner-ready set.

Two independent helpers cover the two input shapes:

- ``compress_hits`` shrinks retrieval hits (already ranked by the RAG layer).
- ``compress_documents`` shrinks raw search documents (scored by search).

Both preserve input order (ranking is trusted), drop empty text, deduplicate
by canonical text / URL, and cap the result. Callers can also cap the length
of each snippet so downstream prompts stay predictable.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document, RetrievalHit
from insightforge.research._text import clean_snippet, normalize

logger = get_logger(__name__)

DEFAULT_MAX_ITEMS = 8
DEFAULT_SNIPPET_CHARS = 240


def compress_hits(
    hits: Sequence[RetrievalHit],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    min_score: float = 0.0,
) -> list[RetrievalHit]:
    """Return a trimmed, deduplicated slice of ``hits``.

    A hit is kept when its text is non-empty, its score meets ``min_score``,
    and its normalized text has not been seen yet. The hit's ``text`` is
    replaced with a truncated snippet so oversized chunks do not bloat later
    prompts. Original order is preserved.
    """

    if max_items <= 0:
        return []

    seen: set[str] = set()
    kept: list[RetrievalHit] = []

    for hit in hits:
        if hit.score < min_score:
            continue
        cleaned = normalize(hit.text)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        snippet = clean_snippet(cleaned, limit=snippet_chars)
        kept.append(hit.model_copy(update={"text": snippet}))
        if len(kept) >= max_items:
            break

    logger.info(
        "compress_hits input=%d kept=%d max=%d",
        len(hits),
        len(kept),
        max_items,
    )
    return kept


def compress_documents(
    documents: Sequence[Document],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> list[Document]:
    """Return a trimmed, deduplicated slice of ``documents``.

    Uses URL as the primary dedupe key (falls back to normalized snippet
    text when a document has no URL). The document's ``snippet`` field is
    replaced with a truncated snippet built from the best available text.
    """

    if max_items <= 0:
        return []

    seen_urls: set[str] = set()
    seen_texts: set[str] = set()
    kept: list[Document] = []

    for doc in documents:
        body = doc.snippet or doc.content
        cleaned = normalize(body)
        if not cleaned:
            continue

        url = (doc.url or "").strip()
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        else:
            key = cleaned.lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)

        snippet = clean_snippet(cleaned, limit=snippet_chars)
        kept.append(doc.model_copy(update={"snippet": snippet}))
        if len(kept) >= max_items:
            break

    logger.info(
        "compress_documents input=%d kept=%d max=%d",
        len(documents),
        len(kept),
        max_items,
    )
    return kept
