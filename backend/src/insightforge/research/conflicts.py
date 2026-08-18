"""Conflict detection — surface disagreements between evidence clusters.

Heuristic: two clusters conflict when their claims share substantial
subject vocabulary but exactly one of them uses a negation cue (``not``,
``no``, ``never``, ``cannot``, ``isn't``, ``fails to``, ...). This catches
the common "X does Y" vs "X does not Y" pattern without an LLM.

An LLM-backed detector can be dropped in later without changing the
returned ``Conflict`` shape.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import Conflict, EvidenceCluster
from insightforge.research._text import content_tokens, jaccard

logger = get_logger(__name__)

# Overlap needed before we even consider two claims to be about the same thing.
DEFAULT_SUBJECT_OVERLAP = 0.3

_NEGATION_TOKENS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "cannot",
        "can't",
        "cant",
        "won't",
        "wont",
        "isn't",
        "isnt",
        "aren't",
        "arent",
        "doesn't",
        "doesnt",
        "don't",
        "dont",
        "wasn't",
        "wasnt",
        "weren't",
        "werent",
        "without",
    }
)

_NEGATION_PHRASES: tuple[str, ...] = (
    "fails to",
    "unable to",
    "does not",
    "did not",
    "cannot",
    "is not",
    "are not",
)


def has_negation(text: str) -> bool:
    """Return True when ``text`` carries a negation cue."""

    lowered = text.lower()
    if any(phrase in lowered for phrase in _NEGATION_PHRASES):
        return True
    tokens = set(lowered.replace("'", "").split())
    return bool(tokens & _NEGATION_TOKENS)


def _sources_of(cluster: EvidenceCluster) -> list[str]:
    return [item.source_id for item in cluster.evidence if item.source_id]


def detect_conflicts(
    clusters: Sequence[EvidenceCluster],
    *,
    subject_overlap: float = DEFAULT_SUBJECT_OVERLAP,
) -> list[Conflict]:
    """Return unordered pairs of clusters that appear to disagree.

    Two clusters conflict when their content tokens overlap by at least
    ``subject_overlap`` (Jaccard) and exactly one of the claims contains a
    negation cue.
    """

    if len(clusters) < 2:
        return []

    prepared: list[tuple[EvidenceCluster, set[str], bool]] = [
        (cluster, content_tokens(cluster.claim), has_negation(cluster.claim))
        for cluster in clusters
    ]

    conflicts: list[Conflict] = []
    for i in range(len(prepared)):
        left_cluster, left_tokens, left_neg = prepared[i]
        if not left_tokens:
            continue
        for j in range(i + 1, len(prepared)):
            right_cluster, right_tokens, right_neg = prepared[j]
            if not right_tokens:
                continue
            if left_neg == right_neg:
                continue
            similarity = jaccard(left_tokens, right_tokens)
            if similarity < subject_overlap:
                continue
            reason = f"opposing polarity on shared subject (token overlap={similarity:.2f})"
            conflicts.append(
                Conflict(
                    claim_a=left_cluster.claim,
                    claim_b=right_cluster.claim,
                    sources_a=_sources_of(left_cluster),
                    sources_b=_sources_of(right_cluster),
                    reason=reason,
                )
            )

    logger.info(
        "detect_conflicts clusters=%d conflicts=%d",
        len(clusters),
        len(conflicts),
    )
    for conflict in conflicts:
        logger.debug(
            "conflict a=%r b=%r reason=%s",
            conflict.claim_a[:80],
            conflict.claim_b[:80],
            conflict.reason,
        )
    return conflicts
