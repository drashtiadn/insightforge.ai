"""Shared helpers for the four RAG quality metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from insightforge.domain.models import EvaluationReport, EvaluationSample, MetricScore
from insightforge.infrastructure.retrieval.tokenize import tokenize
from insightforge.shared.enums import EvaluationBackend, EvaluationMetric

# Library score keys → our four metric names.
_SCORE_ALIASES: dict[str, EvaluationMetric] = {
    "faithfulness": EvaluationMetric.FAITHFULNESS,
    "relevancy": EvaluationMetric.RELEVANCY,
    "answer_relevancy": EvaluationMetric.RELEVANCY,
    "answer_relevance": EvaluationMetric.RELEVANCY,
    "response_relevancy": EvaluationMetric.RELEVANCY,
    "recall": EvaluationMetric.RECALL,
    "context_recall": EvaluationMetric.RECALL,
    "contextual_recall": EvaluationMetric.RECALL,
    "precision": EvaluationMetric.PRECISION,
    "context_precision": EvaluationMetric.PRECISION,
    "contextual_precision": EvaluationMetric.PRECISION,
    "llm_context_precision_without_reference": EvaluationMetric.PRECISION,
    "llm_context_recall": EvaluationMetric.RECALL,
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def clamp_unit(value: float | None) -> float:
    """Clamp a numeric score into ``[0, 1]``. Non-finite values become 0."""

    if value is None:
        return 0.0
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return max(0.0, min(1.0, number))


def token_set(text: str) -> set[str]:
    """Unique lowercased tokens for overlap metrics."""

    return set(tokenize(text))


def coverage(have: set[str], need: set[str]) -> float:
    """Fraction of ``need`` tokens that appear in ``have``."""

    if not need:
        return 0.0
    return len(have & need) / len(need)


def token_f1(left: set[str], right: set[str]) -> float:
    """Token-set F1 between two bags of terms."""

    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into claim-like sentences, skipping tiny fragments."""

    cleaned = text.strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]
    claims = [part for part in parts if len(token_set(part)) >= 3]
    if claims:
        return claims
    return [cleaned]


def normalize_backend_scores(raw: Mapping[str, Any]) -> dict[EvaluationMetric, float]:
    """Map library-specific keys onto ``EvaluationMetric`` with unit scores."""

    mapped: dict[EvaluationMetric, float] = {}
    for key, value in raw.items():
        metric = _SCORE_ALIASES.get(str(key).strip().lower())
        if metric is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        mapped[metric] = clamp_unit(float(value))
    return mapped


def build_report(
    sample: EvaluationSample,
    *,
    backend: EvaluationBackend,
    scores: Mapping[str, Any],
    reasons: Mapping[EvaluationMetric, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Build a complete four-metric report from a backend score map."""

    mapped = normalize_backend_scores(scores)
    extra = dict(metadata or {})
    extra["raw_scores"] = {str(key): value for key, value in scores.items()}
    metrics: list[MetricScore] = []
    for name in EvaluationMetric:
        reason = ""
        if reasons is not None and name in reasons:
            reason = reasons[name]
        elif name not in mapped:
            reason = "metric was not returned by the backend"
        metrics.append(
            MetricScore(
                name=name,
                score=mapped.get(name, 0.0),
                reason=reason,
            )
        )
    overall = sum(item.score for item in metrics) / len(metrics)
    return EvaluationReport(
        query=sample.query,
        backend=backend,
        metrics=metrics,
        overall=clamp_unit(overall),
        context_count=len(sample.contexts),
        ground_truth_used=bool(sample.ground_truth and sample.ground_truth.strip()),
        metadata=extra,
    )


def append_evaluation_section(markdown: str, report: EvaluationReport) -> str:
    """Append (or insert before Errors) the evaluation section."""

    section = report.to_markdown().strip()
    body = markdown.rstrip()
    if not body:
        return section + "\n"
    marker = "\n## Errors\n"
    if marker in markdown:
        head, tail = markdown.split(marker, 1)
        return f"{head.rstrip()}\n\n{section}\n{marker}{tail.lstrip()}"
    return f"{body}\n\n{section}\n"
