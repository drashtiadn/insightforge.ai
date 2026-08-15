"""Okapi BM25 lexical index (no extra dependency)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from typing import Any

from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit
from insightforge.infrastructure.retrieval.tokenize import tokenize
from insightforge.infrastructure.vectorstores.helpers import matches_filters
from insightforge.shared.enums import RetrievalMode

logger = get_logger(__name__)

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass
class _Document:
    id: str
    text: str
    tokens: list[str]
    tf: Counter[str]
    length: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """In-process BM25 index kept in sync with indexed retrieval records.

    Uses Lucene-style IDF so rare and common terms stay non-negative:
    ``log(1 + (N - df + 0.5) / (df + 0.5))``.
    """

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self._k1 = k1
        self._b = b
        self._docs: dict[str, _Document] = {}
        self._df: Counter[str] = Counter()
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    def upsert(
        self,
        *,
        record_id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert or replace one document."""

        cleaned_id = record_id.strip()
        if not cleaned_id:
            return
        self._remove_unlocked(cleaned_id)
        tokens = tokenize(text)
        doc = _Document(
            id=cleaned_id,
            text=text,
            tokens=tokens,
            tf=Counter(tokens),
            length=len(tokens),
            metadata=dict(metadata or {}),
        )
        self._docs[cleaned_id] = doc
        self._total_length += doc.length
        for term in doc.tf:
            self._df[term] += 1

    def delete(self, record_ids: Sequence[str]) -> int:
        """Remove documents by id. Returns how many existed."""

        removed = 0
        for record_id in record_ids:
            if self._remove_unlocked(record_id.strip()):
                removed += 1
        return removed

    def delete_matching(self, filters: Mapping[str, Any]) -> int:
        """Remove documents whose metadata matches ``filters``."""

        to_drop = [doc.id for doc in self._docs.values() if matches_filters(doc.metadata, filters)]
        return self.delete(to_drop)

    def clear(self) -> None:
        self._docs.clear()
        self._df.clear()
        self._total_length = 0

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalHit]:
        """Rank documents for ``query``. Empty corpus or query tokens → []."""

        if limit < 1:
            return []
        query_terms = tokenize(query)
        if not query_terms or not self._docs:
            return []

        avgdl = self._avg_dl()
        n_docs = len(self._docs)
        unique_terms = list(dict.fromkeys(query_terms))
        idf = {term: self._idf(term, n_docs) for term in unique_terms}

        hits: list[RetrievalHit] = []
        for doc in self._docs.values():
            if not matches_filters(doc.metadata, filters):
                continue
            score = 0.0
            for term in unique_terms:
                tf = doc.tf.get(term, 0)
                if tf <= 0:
                    continue
                denom = tf + self._k1 * (1.0 - self._b + self._b * doc.length / avgdl)
                score += idf[term] * (tf * (self._k1 + 1.0)) / denom
            if score_threshold is not None and score < score_threshold:
                continue
            if score <= 0.0:
                continue
            hits.append(
                RetrievalHit(
                    id=doc.id,
                    score=score,
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    mode=RetrievalMode.BM25,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        ranked = hits[:limit]
        for index, hit in enumerate(ranked, start=1):
            hit.bm25_rank = index
        logger.debug(
            "bm25 search hits=%d limit=%d corpus=%d",
            len(ranked),
            limit,
            n_docs,
            extra={"hits": len(ranked), "corpus": n_docs},
        )
        return ranked

    def _avg_dl(self) -> float:
        if not self._docs:
            return 1.0
        return max(1.0, self._total_length / len(self._docs))

    def _idf(self, term: str, n_docs: int) -> float:
        df = self._df.get(term, 0)
        return log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def _remove_unlocked(self, record_id: str) -> bool:
        doc = self._docs.pop(record_id, None)
        if doc is None:
            return False
        self._total_length -= doc.length
        for term in doc.tf:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        return True
