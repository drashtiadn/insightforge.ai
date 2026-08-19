"""Domain models for RAG report evaluation (Phase 7.1).

These models are the contract between evaluation backends (RAGAS, DeepEval,
heuristic) and the research pipeline. They stay free of infrastructure so
API schemas and graph state can serialize them without extra adapters.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from insightforge.shared.enums import EvaluationBackend, EvaluationMetric

__all__ = [
    "EvaluationReport",
    "EvaluationSample",
    "MetricScore",
]


class EvaluationSample(BaseModel):
    """One query/answer/context bundle to score.

    ``ground_truth`` is optional. Live research runs usually omit it; recall
    then uses ``answer`` as the reference (a standard automatic proxy).
    """

    query: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str | None = None

    @property
    def reference(self) -> str:
        """Preferred text for context-recall: ground truth, else the answer."""

        if self.ground_truth and self.ground_truth.strip():
            return self.ground_truth.strip()
        return self.answer.strip()


class MetricScore(BaseModel):
    """A single 0..1 quality score with a short debug reason."""

    name: EvaluationMetric
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class EvaluationReport(BaseModel):
    """Automatic quality report attached to a research answer."""

    query: str
    backend: EvaluationBackend
    metrics: list[MetricScore] = Field(default_factory=list)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    context_count: int = 0
    ground_truth_used: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def score_for(self, name: EvaluationMetric) -> float | None:
        """Return the score for ``name``, or ``None`` if it was not computed."""

        for metric in self.metrics:
            if metric.name is name:
                return metric.score
        return None

    def to_markdown(self) -> str:
        """Render a Markdown section suitable for appending to a report."""

        lines = [
            "## Evaluation",
            "",
            (
                f"Automatic quality scores (backend: `{self.backend.value}`, "
                f"contexts: {self.context_count})."
            ),
            "",
        ]
        for metric in self.metrics:
            reason = f" — {metric.reason}" if metric.reason else ""
            label = metric.name.value.replace("_", " ").title()
            lines.append(f"- **{label}**: {metric.score:.2f}{reason}")
        lines.append(f"- **Overall**: {self.overall:.2f}")
        lines.append("")
        return "\n".join(lines)
