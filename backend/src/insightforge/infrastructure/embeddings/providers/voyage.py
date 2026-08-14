"""Voyage AI embedding provider (https://docs.voyageai.com)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import EmbeddingResult
from insightforge.infrastructure.embeddings.base import EmbeddingProvider
from insightforge.infrastructure.embeddings.http import (
    batched,
    raise_for_status,
    require_texts,
)
from insightforge.shared.enums import EmbeddingInputType, EmbeddingProviderHint

logger = get_logger(__name__)

VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Remote embeddings via Voyage AI REST API."""

    name = EmbeddingProviderHint.VOYAGE

    def __init__(
        self,
        client: httpx.Client,
        *,
        api_key: str | None = None,
        model: str = "voyage-3.5",
        dimensions: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self._client = client
        self._api_key = (api_key or "").strip() or None
        self._model = (model or "").strip() or "voyage-3.5"
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def model(self) -> str:
        return self._model

    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> EmbeddingResult:
        cleaned = require_texts(texts)
        if not cleaned:
            return EmbeddingResult(
                vectors=[],
                model=self._model,
                provider=self.name,
                dimensions=0,
                input_type=input_type,
                usage_tokens=0,
                metadata={"backend": "voyage"},
            )

        if not self._api_key:
            raise ExternalServiceError(
                "voyage embeddings require VOYAGE_API_KEY",
                details={"provider": self.name.value},
            )

        logger.info(
            "voyage embed started count=%d model=%s input_type=%s batch_size=%d",
            len(cleaned),
            self._model,
            input_type.value,
            self._batch_size,
            extra={
                "provider": self.name.value,
                "model": self._model,
                "count": len(cleaned),
                "input_type": input_type.value,
            },
        )

        vectors: list[list[float]] = []
        total_tokens = 0
        for batch in batched(cleaned, self._batch_size):
            batch_vectors, tokens = self._embed_batch(batch, input_type=input_type)
            vectors.extend(batch_vectors)
            if tokens is not None:
                total_tokens += tokens

        if len(vectors) != len(cleaned):
            raise ExternalServiceError(
                "voyage returned unexpected embedding count",
                details={
                    "provider": self.name.value,
                    "expected": len(cleaned),
                    "actual": len(vectors),
                },
            )

        dimensions = len(vectors[0]) if vectors else 0
        logger.info(
            "voyage embed finished count=%d dims=%d tokens=%s",
            len(vectors),
            dimensions,
            total_tokens or None,
            extra={
                "provider": self.name.value,
                "count": len(vectors),
                "dimensions": dimensions,
            },
        )
        return EmbeddingResult(
            vectors=vectors,
            model=self._model,
            provider=self.name,
            dimensions=dimensions,
            input_type=input_type,
            usage_tokens=total_tokens or None,
            metadata={"backend": "voyage"},
        )

    def _embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType,
    ) -> tuple[list[list[float]], int | None]:
        payload: dict[str, Any] = {
            "input": list(texts),
            "model": self._model,
            "input_type": input_type.value,
        }
        if self._dimensions is not None:
            payload["output_dimension"] = self._dimensions

        response = self._client.post(
            VOYAGE_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        raise_for_status(response, provider="voyage")

        body = response.json()
        data = body.get("data")
        if not isinstance(data, list):
            raise ExternalServiceError(
                "voyage embeddings response missing data",
                details={"provider": self.name.value},
            )

        indexed: list[tuple[int, list[float]]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            index = item.get("index")
            if not isinstance(index, int):
                index = len(indexed)
            indexed.append((index, [float(value) for value in embedding]))

        indexed.sort(key=lambda pair: pair[0])
        vectors = [vector for _, vector in indexed]

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        tokens_raw = usage.get("total_tokens") if isinstance(usage, dict) else None
        tokens = int(tokens_raw) if isinstance(tokens_raw, int) else None
        return vectors, tokens
