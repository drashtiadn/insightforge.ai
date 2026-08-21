"""Phase 7 evaluation prompts and metric definitions.

Runtime scoring lives in ``insightforge.infrastructure.evaluation``.
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

JUDGE_SYSTEM = """\
You are the InsightForge quality judge. Decide whether the research answer is \
trustworthy enough to ship. Return JSON only:
{"passed": true,
 "confidence": 0.0,
 "issues": ["<short issue>", "..."],
 "revision_hint": "<what to fix if passed is false>"}
confidence must be a number between 0 and 1. Set passed to true only when the \
answer is faithful to the context, relevant to the query, and your confidence \
is at least the given threshold. If passed is false, give a concrete revision_hint \
the reasoner can follow. Do not invent sources.\
"""

JUDGE_USER = """\
Query: {query}

Answer:
{answer}

Retrieved context:
{contexts}

Automatic metrics:
{metrics}
Overall: {overall}
Confidence threshold: {threshold}
"""

__all__ = ["JUDGE_SYSTEM", "JUDGE_USER", "METRIC_HELP"]
