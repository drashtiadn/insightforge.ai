"""Deduplicate search documents by normalized URL."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from insightforge.core.logging import get_logger
from insightforge.domain.models import Document

logger = get_logger(__name__)

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
    }
)


def normalize_url(url: str) -> str:
    """Canonicalize a URL for comparison (host, path, non-tracking query)."""

    raw = url.strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    port = parsed.port
    netloc = host
    if port and port not in {80, 443}:
        netloc = f"{host}:{port}"

    path = parsed.path.rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(query_pairs))
    return urlunparse((scheme, netloc, path, "", query, ""))


def dedupe_documents(documents: list[Document]) -> list[Document]:
    """Drop duplicate URLs, keeping the higher-scored (or first) document."""

    best: dict[str, Document] = {}
    order: list[str] = []

    for doc in documents:
        key = normalize_url(doc.url) or doc.url.strip().lower()
        if not key:
            continue
        existing = best.get(key)
        if existing is None:
            best[key] = doc
            order.append(key)
            continue
        if _prefer(doc, existing):
            best[key] = doc

    result = [best[key] for key in order]
    removed = len(documents) - len(result)
    if removed:
        logger.info(
            "deduped documents removed=%d kept=%d",
            removed,
            len(result),
            extra={"removed": removed, "kept": len(result)},
        )
    return result


def _prefer(candidate: Document, current: Document) -> bool:
    """True when ``candidate`` should replace ``current``."""

    cand_score = candidate.score
    curr_score = current.score
    if cand_score is not None and curr_score is not None:
        return cand_score > curr_score
    if cand_score is not None and curr_score is None:
        return True
    # Prefer richer snippets when scores are tied/missing.
    return len(candidate.snippet) > len(current.snippet)
