"""BGE reranker (BAAI CrossEncoder checkpoints)."""

from __future__ import annotations

from insightforge.infrastructure.rerankers.providers.local import LocalReranker
from insightforge.shared.enums import RerankerHint


class BGEReranker(LocalReranker):
    """Local BGE reranker (``BAAI/bge-reranker-base`` by default)."""

    name = RerankerHint.BGE
    default_model = "BAAI/bge-reranker-base"
