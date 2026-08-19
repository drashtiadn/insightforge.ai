"""Phase 7.1 evaluation metric definitions.

Runtime scoring lives in ``insightforge.infrastructure.evaluation``.
LLM-as-judge retry / confidence gates are Phase 7.2.
"""

from insightforge.shared.enums import EvaluationMetric

METRIC_HELP = {
    EvaluationMetric.FAITHFULNESS: (
        "Fraction of answer claims that are supported by retrieved context."
    ),
    EvaluationMetric.RELEVANCY: ("How well the answer addresses the original query."),
    EvaluationMetric.RECALL: (
        "How completely retrieved context covers the reference (ground truth or answer)."
    ),
    EvaluationMetric.PRECISION: (
        "How much of the retrieved context is relevant to the query, with higher rank preferred."
    ),
}

__all__ = ["METRIC_HELP"]
