"""Evidence aggregation — cluster corroborating snippets into claims.

The deterministic clusterer here groups snippets whose content-token overlap
(Jaccard similarity) exceeds a threshold. The strongest snippet in a cluster
becomes the ``claim``. Confidence-weighted scoring lets later stages pick
the most supported claims first.

This is deliberately simple: no LLM, no embeddings. Phase 6.2's LLM-backed
reasoner can replace the clustering internals without changing the returned
``EvidenceCluster`` shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Document,
    Evidence,
    EvidenceCluster,
    RetrievalHit,
)
from insightforge.research._text import (
    clean_snippet,
    content_tokens,
    jaccard,
    normalize,
)
from insightforge.research.conflicts import has_negation

logger = get_logger(__name__)

# Snippets are considered to support the same claim when their content-token
# Jaccard similarity is at least this high. Tuned for short retrieval snippets.
DEFAULT_SIMILARITY_THRESHOLD = 0.3
# Prevent one over-broad claim from swallowing every snippet.
DEFAULT_MAX_CLAIMS = 8
DEFAULT_CLAIM_CHARS = 240


@dataclass
class _WorkingCluster:
    """Mutable clustering bucket used only during aggregation."""

    claim_text: str
    claim_tokens: set[str]
    claim_score: float
    tokens: set[str]
    negated: bool
    members: list[Evidence] = field(default_factory=list)


def _evidence_key(evidence: Evidence) -> str:
    """Stable identity so the same source is not clustered twice."""

    if evidence.url:
        return f"url::{evidence.url.strip().lower()}"
    if evidence.title:
        return f"title::{evidence.title.strip().lower()}"
    snippet = normalize(evidence.snippet).lower()
    if snippet:
        return f"text::{snippet}"
    return f"source::{evidence.source_id.strip().lower()}"


def _evidence_from_hit(hit: RetrievalHit) -> Evidence:
    metadata = dict(hit.metadata)
    url = metadata.get("url")
    title = metadata.get("title")
    url_text = str(url).strip() if url else ""
    # Prefer URL so hit/document pairs for the same page share one source id.
    source_id = url_text or hit.id
    return Evidence(
        source_id=source_id,
        snippet=clean_snippet(hit.text),
        score=float(hit.score),
        url=url_text or None,
        title=str(title) if title else None,
        metadata=metadata,
    )


def _evidence_from_document(doc: Document) -> Evidence:
    body = doc.snippet or doc.content
    return Evidence(
        source_id=doc.url or doc.title,
        snippet=clean_snippet(body),
        score=float(doc.score or 0.0),
        url=doc.url or None,
        title=doc.title or None,
        metadata={"provider": doc.provider.value, **dict(doc.metadata)},
    )


def _add_to_cluster(cluster: _WorkingCluster, evidence: Evidence, tokens: set[str]) -> None:
    cluster.members.append(evidence)
    cluster.tokens |= tokens
    # Promote to claim when this snippet is stronger than the current claim.
    if evidence.score > cluster.claim_score and evidence.snippet:
        cluster.claim_text = evidence.snippet
        cluster.claim_tokens = tokens
        cluster.claim_score = evidence.score


def _cluster_score(cluster: _WorkingCluster) -> float:
    if not cluster.members:
        return 0.0
    total = sum(member.score for member in cluster.members)
    # Reward corroboration: n supporters scale via log1p, capped at 1.0.
    support = min(1.0, 0.4 + 0.15 * (len(cluster.members) - 1))
    return min(1.0, (total / len(cluster.members)) * support)


def aggregate_evidence(
    *,
    hits: Sequence[RetrievalHit] | None = None,
    documents: Sequence[Document] | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    claim_chars: int = DEFAULT_CLAIM_CHARS,
) -> list[EvidenceCluster]:
    """Group snippets that support the same claim.

    ``hits`` are consumed first (typically higher signal from the RAG layer);
    document snippets fall in afterwards. Order of returned clusters matches
    the order in which their claim was first observed, so upstream ranking
    is preserved.
    """

    pool: list[Evidence] = []
    seen: set[str] = set()
    for hit in hits or []:
        evidence = _evidence_from_hit(hit)
        key = _evidence_key(evidence)
        if key in seen:
            continue
        seen.add(key)
        pool.append(evidence)
    for doc in documents or []:
        evidence = _evidence_from_document(doc)
        key = _evidence_key(evidence)
        if key in seen:
            continue
        seen.add(key)
        pool.append(evidence)

    working: list[_WorkingCluster] = []
    for evidence in pool:
        if not evidence.snippet:
            continue
        tokens = content_tokens(evidence.snippet)
        if not tokens:
            continue

        negated = has_negation(evidence.snippet)
        best: _WorkingCluster | None = None
        best_score = 0.0
        for cluster in working:
            # Never fold opposing polarities into one cluster — the conflict
            # detector needs to see them as separate claims.
            if cluster.negated != negated:
                continue
            similarity = jaccard(tokens, cluster.tokens)
            if similarity >= similarity_threshold and similarity > best_score:
                best = cluster
                best_score = similarity

        if best is not None:
            _add_to_cluster(best, evidence, tokens)
            continue

        working.append(
            _WorkingCluster(
                claim_text=evidence.snippet,
                claim_tokens=tokens,
                claim_score=evidence.score,
                tokens=set(tokens),
                negated=negated,
                members=[evidence],
            )
        )

    working.sort(key=_cluster_score, reverse=True)
    working = working[:max_claims]

    clusters = [
        EvidenceCluster(
            claim=normalize(cluster.claim_text)[:claim_chars],
            evidence=list(cluster.members),
            score=round(_cluster_score(cluster), 4),
        )
        for cluster in working
    ]

    logger.info(
        "aggregate_evidence hits=%d documents=%d clusters=%d",
        len(hits or []),
        len(documents or []),
        len(clusters),
    )
    for finished in clusters:
        logger.debug(
            "cluster claim=%r score=%.2f members=%d",
            finished.claim[:80],
            finished.score,
            len(finished.evidence),
        )
    return clusters
