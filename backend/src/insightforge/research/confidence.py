"""Confidence scoring — bounded scalar signals used across research agents.

Confidence is intentionally simple and explainable. Each factor contributes
a bounded amount, and the final score is clipped to ``[0.0, 1.0]``. Both
scorers return ``0.0`` when no evidence is present so callers can safely
gate on it.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import Conflict, EvidenceCluster

logger = get_logger(__name__)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def reasoning_confidence(
    clusters: Sequence[EvidenceCluster],
    conflicts: Sequence[Conflict],
    *,
    source_count: int,
) -> float:
    """Score how much we trust the reasoner's synthesized answer.

    Weights (chosen for readability, not tuning):
    - baseline 0.20 when any evidence exists
    - +0.08 per corroborated cluster (up to 0.40)
    - +avg(cluster.score) * 0.30 for direct evidence quality
    - -0.12 per detected conflict (up to -0.30)
    - +0.05 per additional source beyond the first (up to 0.10)
    """

    if not clusters and source_count == 0:
        return 0.0

    score = 0.20
    corroborated = min(0.40, 0.08 * len(clusters))
    quality = 0.0
    if clusters:
        quality = 0.30 * (sum(cluster.score for cluster in clusters) / len(clusters))
    conflict_penalty = min(0.30, 0.12 * len(conflicts))
    source_bonus = min(0.10, 0.05 * max(source_count - 1, 0))

    total = score + corroborated + quality + source_bonus - conflict_penalty
    result = round(_clip(total), 4)
    logger.debug(
        "reasoning_confidence clusters=%d conflicts=%d sources=%d score=%.3f",
        len(clusters),
        len(conflicts),
        source_count,
        result,
    )
    return result


def reflection_confidence(
    *,
    is_sufficient: bool,
    source_count: int,
    conflict_count: int,
    reasoning: float,
) -> float:
    """Score how confidently reflection endorses (or rejects) the state.

    Blends the reasoner's own confidence with structural signals so a
    reflection with a strong reasoner but few sources still lands low.
    """

    base = 0.25 + 0.5 * reasoning
    coverage = min(0.20, 0.05 * source_count)
    conflict_penalty = min(0.30, 0.10 * conflict_count)
    verdict_bonus = 0.10 if is_sufficient else -0.10

    total = base + coverage + verdict_bonus - conflict_penalty
    result = round(_clip(total), 4)
    logger.debug(
        "reflection_confidence sufficient=%s sources=%d conflicts=%d reasoning=%.2f score=%.3f",
        is_sufficient,
        source_count,
        conflict_count,
        reasoning,
        result,
    )
    return result
