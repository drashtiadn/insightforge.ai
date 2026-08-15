"""Jina AI reranker (https://api.jina.ai/v1/rerank)."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit
from insightforge.infrastructure.rerankers.base import RerankerProvider
from insightforge.infrastructure.rerankers.http import (
    apply_scores,
    raise_for_status,
    require_query,
)
from insightforge.shared.enums import RerankerHint

logger = get_logger(__name__)

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker(RerankerProvider):
    """Remote reranking via the Jina Search Foundation API."""

    name = RerankerHint.JINA

    def __init__(
        self,
        client: httpx.Client,
        *,
        api_key: str | None = None,
        model: str = "jina-reranker-v2-base-multilingual",
    ) -> None:
        self._client = client
        self._api_key = (api_key or "").strip() or None
        self._model = (model or "").strip() or "jina-reranker-v2-base-multilingual"

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def model(self) -> str:
        return self._model

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_n: int | None = None,
    ) -> list[RetrievalHit]:
        cleaned = require_query(query)
        if not hits:
            return []
        if not self._api_key:
            raise ExternalServiceError(
                "jina rerank requires JINA_API_KEY",
                details={"provider": self.name.value},
            )

        limit = top_n if top_n is not None else len(hits)
        if limit < 1:
            return []

        logger.info(
            "jina rerank started model=%s candidates=%d top_n=%d",
            self._model,
            len(hits),
            limit,
            extra={
                "provider": self.name.value,
                "model": self._model,
                "candidates": len(hits),
            },
        )

        documents = [hit.text if hit.text.strip() else " " for hit in hits]
        response = self._client.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "query": cleaned,
                "documents": documents,
                "top_n": limit,
                "return_documents": False,
            },
        )
        raise_for_status(response, provider="jina")

        body = response.json()
        results = body.get("results")
        if not isinstance(results, list):
            raise ExternalServiceError(
                "jina rerank response missing results",
                details={"provider": self.name.value},
            )

        scores = [0.0] * len(hits)
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(hits):
                continue
            raw_score = item.get("relevance_score", item.get("score"))
            scores[index] = float(raw_score) if isinstance(raw_score, int | float) else 0.0
            seen.add(index)

        if not seen:
            raise ExternalServiceError(
                "jina rerank returned no usable results",
                details={"provider": self.name.value},
            )

        ranked = apply_scores(hits, scores, top_n=limit)
        logger.info(
            "jina rerank finished count=%d",
            len(ranked),
            extra={"provider": self.name.value, "count": len(ranked)},
        )
        return ranked
