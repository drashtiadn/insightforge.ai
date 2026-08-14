"""Generic Cross Encoder reranker (MS MARCO MiniLM by default)."""

from __future__ import annotations

from insightforge.infrastructure.rerankers.providers.local import LocalReranker, RerankFn
from insightforge.shared.enums import RerankerHint


class CrossEncoderReranker(LocalReranker):
    """Local pairwise CrossEncoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``)."""

    name = RerankerHint.CROSS_ENCODER
    default_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"


__all__ = ["CrossEncoderReranker", "RerankFn"]
