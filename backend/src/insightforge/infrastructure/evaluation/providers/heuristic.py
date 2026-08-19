"""Lexical approximations of RAGAS / DeepEval metrics.

Used when RAGAS and DeepEval are not installed, and as a fallback when those
backends fail. Scores are deterministic and require no API keys.
"""

from __future__ import annotations

from insightforge.core.logging import get_logger
from insightforge.domain.models import EvaluationReport, EvaluationSample
from insightforge.infrastructure.evaluation.base import Evaluator
from insightforge.infrastructure.evaluation.metrics import (
    build_report,
    coverage,
    split_sentences,
    token_f1,
    token_set,
)
from insightforge.shared.enums import EvaluationBackend, EvaluationMetric

logger = get_logger(__name__)

# A context chunk is "relevant" when it overlaps the query at least this much.
_RELEVANT_F1 = 0.12
# Or when this fraction of its tokens appear in the query/answer union.
_RELEVANT_COVERAGE = 0.2


class HeuristicEvaluator(Evaluator):
    """Token-overlap implementations of the four standard RAG metrics."""

    name = EvaluationBackend.HEURISTIC

    def evaluate(self, sample: EvaluationSample) -> EvaluationReport:
        answer_tokens = token_set(sample.answer)
        query_tokens = token_set(sample.query)
        context_tokens = token_set(" ".join(sample.contexts))
        reference_tokens = token_set(sample.reference)

        faithfulness, faith_reason = _faithfulness(sample, context_tokens)
        relevancy, rel_reason = _relevancy(query_tokens, answer_tokens)
        recall, recall_reason = _recall(context_tokens, reference_tokens, sample)
        precision, prec_reason = _precision(sample, query_tokens, answer_tokens)

        logger.info(
            "heuristic evaluation faithfulness=%.3f relevancy=%.3f recall=%.3f precision=%.3f",
            faithfulness,
            relevancy,
            recall,
            precision,
            extra={
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "recall": recall,
                "precision": precision,
                "contexts": len(sample.contexts),
            },
        )
        return build_report(
            sample,
            backend=self.name,
            scores={
                EvaluationMetric.FAITHFULNESS.value: faithfulness,
                EvaluationMetric.RELEVANCY.value: relevancy,
                EvaluationMetric.RECALL.value: recall,
                EvaluationMetric.PRECISION.value: precision,
            },
            reasons={
                EvaluationMetric.FAITHFULNESS: faith_reason,
                EvaluationMetric.RELEVANCY: rel_reason,
                EvaluationMetric.RECALL: recall_reason,
                EvaluationMetric.PRECISION: prec_reason,
            },
        )


def _faithfulness(sample: EvaluationSample, context_tokens: set[str]) -> tuple[float, str]:
    claims = split_sentences(sample.answer)
    if not claims:
        return 0.0, "no answer claims to ground"
    if not context_tokens:
        return 0.0, "no retrieved context to ground the answer"
    supported = 0
    total = 0
    for claim in claims:
        tokens = token_set(claim)
        if not tokens:
            continue
        total += 1
        if coverage(context_tokens, tokens) >= 0.5:
            supported += 1
    if total == 0:
        return 0.0, "no answer claims to ground"
    score = supported / total
    return score, f"{supported}/{total} answer sentences are grounded in retrieved context"


def _relevancy(query_tokens: set[str], answer_tokens: set[str]) -> tuple[float, str]:
    if not query_tokens:
        return 0.0, "empty query"
    if not answer_tokens:
        return 0.0, "empty answer"
    score = token_f1(query_tokens, answer_tokens)
    covered = coverage(answer_tokens, query_tokens)
    # Blend F1 with query-term coverage so long answers are not punished as hard.
    blended = 0.5 * score + 0.5 * covered
    return blended, "answer overlap with the query"


def _recall(
    context_tokens: set[str],
    reference_tokens: set[str],
    sample: EvaluationSample,
) -> tuple[float, str]:
    if not reference_tokens:
        return 0.0, "no reference text for recall"
    if not context_tokens:
        return 0.0, "no retrieved context"
    score = coverage(context_tokens, reference_tokens)
    source = "ground truth" if sample.ground_truth else "answer"
    return score, f"retrieved context covers {source} terms"


def _precision(
    sample: EvaluationSample,
    query_tokens: set[str],
    answer_tokens: set[str],
) -> tuple[float, str]:
    if not sample.contexts:
        return 0.0, "no retrieved context"
    union = query_tokens | answer_tokens
    relevant_count = 0
    ranked_precisions: list[float] = []
    for index, chunk in enumerate(sample.contexts, start=1):
        chunk_tokens = token_set(chunk)
        relevant = _chunk_is_relevant(chunk_tokens, query_tokens, union)
        if relevant:
            relevant_count += 1
            ranked_precisions.append(relevant_count / index)
    if not ranked_precisions:
        return 0.0, "0 retrieved chunks are relevant to the query"
    score = sum(ranked_precisions) / len(ranked_precisions)
    return (
        score,
        f"{relevant_count}/{len(sample.contexts)} retrieved chunks are relevant to the query",
    )


def _chunk_is_relevant(
    chunk_tokens: set[str],
    query_tokens: set[str],
    union: set[str],
) -> bool:
    if not chunk_tokens:
        return False
    if token_f1(query_tokens, chunk_tokens) >= _RELEVANT_F1:
        return True
    return coverage(union, chunk_tokens) >= _RELEVANT_COVERAGE
