"""Citation generation — build a canonical bibliography with claim linking.

``build_citations`` walks every ``EvidenceCluster`` and produces a de-duplicated
list of ``Citation`` objects preserving first-seen order. Retrieval-only
snippets (no URL) become locator-style citations so the report can still
attribute them.

``link_clusters_to_citations`` returns a parallel list mapping each cluster
to the 1-based indexes of the citations that support its claim. The report
generator uses that to attach superscript references to each finding.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import Citation, Evidence, EvidenceCluster

logger = get_logger(__name__)


def _key(evidence: Evidence) -> str:
    """Stable dedupe key for a citation candidate."""

    if evidence.url:
        return f"url::{evidence.url.strip().lower()}"
    if evidence.title:
        return f"title::{evidence.title.strip().lower()}"
    return f"source::{evidence.source_id.strip().lower()}"


def _citation_from_evidence(evidence: Evidence) -> Citation:
    locator: str | None = None
    if not evidence.url:
        # No URL — expose the retrieval source id so readers can trace it.
        locator = f"chunk {evidence.source_id}"
    return Citation(
        url=evidence.url or None,
        title=evidence.title or None,
        site_name=str(evidence.metadata.get("provider"))
        if evidence.metadata.get("provider")
        else None,
        locator=locator,
    )


def build_citations(clusters: Sequence[EvidenceCluster]) -> list[Citation]:
    """Return a de-duplicated list of citations across all clusters."""

    seen: dict[str, int] = {}
    citations: list[Citation] = []

    for cluster in clusters:
        for evidence in cluster.evidence:
            key = _key(evidence)
            if key in seen:
                continue
            seen[key] = len(citations)
            citations.append(_citation_from_evidence(evidence))

    logger.info(
        "build_citations clusters=%d citations=%d",
        len(clusters),
        len(citations),
    )
    return citations


def link_clusters_to_citations(
    clusters: Sequence[EvidenceCluster],
    citations: Sequence[Citation],
) -> list[list[int]]:
    """Return 1-based citation indexes supporting each cluster's claim.

    The mapping honors the order returned by ``build_citations`` so callers
    can render inline references like ``[1, 3]`` directly.
    """

    if not clusters:
        return []

    index_lookup: dict[str, int] = {}
    for index, citation in enumerate(citations, start=1):
        key = _citation_lookup_key(citation)
        index_lookup.setdefault(key, index)

    links: list[list[int]] = []
    for cluster in clusters:
        indexes: list[int] = []
        seen: set[int] = set()
        for evidence in cluster.evidence:
            key = _evidence_lookup_key(evidence)
            found = index_lookup.get(key)
            if found is None or found in seen:
                continue
            seen.add(found)
            indexes.append(found)
        links.append(sorted(indexes))
    return links


def _citation_lookup_key(citation: Citation) -> str:
    if citation.url:
        return f"url::{citation.url.strip().lower()}"
    if citation.title:
        return f"title::{citation.title.strip().lower()}"
    if citation.locator:
        return f"locator::{citation.locator.strip().lower()}"
    return "unknown"


def _evidence_lookup_key(evidence: Evidence) -> str:
    if evidence.url:
        return f"url::{evidence.url.strip().lower()}"
    if evidence.title:
        return f"title::{evidence.title.strip().lower()}"
    return f"locator::chunk {evidence.source_id.strip().lower()}"
