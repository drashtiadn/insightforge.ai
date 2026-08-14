"""Semantic chunker — group sentences until a similarity drop or size limit."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import DocumentChunk, ParsedDocument
from insightforge.infrastructure.document.chunking.base import ChunkConfig, DocumentChunker
from insightforge.infrastructure.document.chunking.helpers import (
    build_chunks,
    jaccard,
    split_sentences,
    tokenize,
)
from insightforge.infrastructure.document.chunking.recursive import split_recursive
from insightforge.shared.enums import ChunkStrategy

logger = get_logger(__name__)

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

# Sentence row: (text, start, end) offsets into the source document.
_Sentence = tuple[str, int, int]


class SemanticChunker(DocumentChunker):
    """Group adjacent sentences that stay on the same topic.

    When ``embed_text`` is provided, adjacent-group cosine similarity is used.
    Otherwise a lexical Jaccard fallback is used (no extra model dependency).

    Group text is always a contiguous slice of the source document so chunk
    ``start`` / ``end`` offsets remain accurate for citation highlighting.
    """

    name = ChunkStrategy.SEMANTIC

    def __init__(
        self,
        config: ChunkConfig | None = None,
        *,
        embed_text: EmbedFn | None = None,
    ) -> None:
        super().__init__(config)
        self._embed_text = embed_text

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.text
        if not text.strip():
            logger.info("semantic chunker skipped empty document")
            return []

        sentences = split_sentences(text)
        if not sentences:
            return []

        vectors = self._embed_sentences([item[0] for item in sentences])
        groups = self._group(text, sentences, vectors)
        chunks = build_chunks(
            document,
            [(group, None) for group in groups],
            strategy=self.name,
            extra_metadata={
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "semantic_threshold": self.config.semantic_threshold,
                "embedder": "custom" if self._embed_text is not None else "lexical",
            },
        )
        logger.info(
            "semantic chunker finished chars=%d sentences=%d chunks=%d embedder=%s",
            len(text),
            len(sentences),
            len(chunks),
            "custom" if self._embed_text is not None else "lexical",
        )
        return chunks

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]] | None:
        if self._embed_text is None:
            return None
        try:
            raw = self._embed_text(sentences)
        except Exception as exc:
            logger.warning("semantic embed failed; falling back to lexical error=%s", exc)
            return None
        vectors = [list(vector) for vector in raw]
        if len(vectors) != len(sentences):
            logger.warning(
                "semantic embed size mismatch got=%d expected=%d; using lexical",
                len(vectors),
                len(sentences),
            )
            return None
        return vectors

    def _group(
        self,
        source: str,
        sentences: list[_Sentence],
        vectors: list[list[float]] | None,
    ) -> list[str]:
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        threshold = self.config.semantic_threshold
        min_keep = max(1, chunk_size // 4)

        groups: list[str] = []
        # Indices into ``sentences`` for the in-progress group.
        current: list[int] = []
        current_vecs: list[list[float]] = []

        def span_text(indices: list[int]) -> str:
            if not indices:
                return ""
            start = sentences[indices[0]][1]
            end = sentences[indices[-1]][2]
            return source[start:end]

        def emit(keep_overlap: bool) -> None:
            nonlocal current, current_vecs
            joined = span_text(current)
            if not joined.strip():
                current = []
                current_vecs = []
                return
            if len(joined) > chunk_size:
                groups.extend(split_recursive(joined, chunk_size=chunk_size, chunk_overlap=overlap))
            else:
                groups.append(joined)
            if keep_overlap and overlap > 0 and current:
                current, current_vecs = _overlap_sentence_indices(
                    current,
                    sentences,
                    current_vecs if vectors is not None else None,
                    overlap,
                    source,
                )
            else:
                current = []
                current_vecs = []

        for index, (sentence, _start, _end) in enumerate(sentences):
            vector = vectors[index] if vectors is not None else None
            if len(sentence) > chunk_size:
                emit(keep_overlap=False)
                groups.extend(
                    split_recursive(sentence, chunk_size=chunk_size, chunk_overlap=overlap)
                )
                continue

            if not current:
                current = [index]
                current_vecs = [vector] if vector is not None else []
                continue

            candidate_len = len(span_text([*current, index]))
            over_size = candidate_len > chunk_size
            similar = True
            if not over_size and len(span_text(current)) >= min_keep:
                similar = _is_similar(
                    [sentences[i][0] for i in current],
                    current_vecs,
                    sentence,
                    vector,
                    threshold,
                )

            if over_size or not similar:
                emit(keep_overlap=True)
                if current and len(span_text([*current, index])) > chunk_size:
                    current = []
                    current_vecs = []
                current.append(index)
                if vector is not None:
                    current_vecs.append(vector)
                continue

            current.append(index)
            if vector is not None:
                current_vecs.append(vector)

        emit(keep_overlap=False)
        return groups


def _is_similar(
    current: list[str],
    current_vecs: list[list[float]],
    next_sentence: str,
    next_vec: list[float] | None,
    threshold: float,
) -> bool:
    if next_vec is not None and current_vecs:
        return _cosine(_mean(current_vecs), next_vec) >= threshold
    return jaccard(tokenize(" ".join(current)), tokenize(next_sentence)) >= threshold


def _overlap_sentence_indices(
    indices: list[int],
    sentences: list[_Sentence],
    vectors: list[list[float]] | None,
    overlap: int,
    source: str,
) -> tuple[list[int], list[list[float]]]:
    """Keep a trailing window of sentences whose source span length ≤ overlap."""

    window: list[int] = []
    window_vecs: list[list[float]] = []
    for position in range(len(indices) - 1, -1, -1):
        candidate = [indices[position], *window]
        start = sentences[candidate[0]][1]
        end = sentences[candidate[-1]][2]
        length = end - start
        if window and length > overlap:
            break
        window = candidate
        if vectors is not None:
            window_vecs = [vectors[position], *window_vecs]
    # Prefer non-empty source text; empty slice should not retain indices.
    if window and not source[sentences[window[0]][1] : sentences[window[-1]][2]].strip():
        return [], []
    return window, window_vecs


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    count = 0
    for vector in vectors:
        if len(vector) != dim:
            continue
        for i, value in enumerate(vector):
            acc[i] += float(value)
        count += 1
    if count == 0:
        return []
    return [value / count for value in acc]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return float(dot / ((norm_left**0.5) * (norm_right**0.5)))
