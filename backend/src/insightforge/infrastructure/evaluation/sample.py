"""Build evaluation samples from pipeline outputs."""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Document,
    EvaluationSample,
    RetrievalHit,
)

logger = get_logger(__name__)


def contexts_from_hits_and_documents(
    hits: Sequence[RetrievalHit],
    documents: Sequence[Document],
) -> list[str]:
    """Prefer retrieval hit text; fall back to document bodies."""

    texts: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        text = (hit.text or "").strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    if texts:
        return texts
    for document in documents:
        text = (document.content or document.snippet or "").strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def build_sample(
    *,
    query: str,
    answer: str,
    contexts: Sequence[str],
    ground_truth: str | None = None,
) -> EvaluationSample:
    """Normalize a sample for scoring."""

    cleaned_contexts = [item.strip() for item in contexts if item and item.strip()]
    truth = ground_truth.strip() if ground_truth and ground_truth.strip() else None
    sample = EvaluationSample(
        query=query.strip(),
        answer=answer.strip(),
        contexts=cleaned_contexts,
        ground_truth=truth,
    )
    logger.debug(
        "evaluation sample query=%r answer_chars=%d contexts=%d has_ground_truth=%s",
        sample.query,
        len(sample.answer),
        len(sample.contexts),
        sample.ground_truth is not None,
    )
    return sample
