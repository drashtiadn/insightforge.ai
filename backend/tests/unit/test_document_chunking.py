"""Tests for Phase 4.3 document chunking."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from insightforge.core.exceptions import ValidationFailedError
from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document import (
    ChunkConfig,
    DocumentParseService,
    MarkdownAwareChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    build_parsers,
    chunk_document,
    parse_chunk_strategy,
)
from insightforge.shared.enums import ChunkStrategy, ContentType


def _doc(text: str, content_type: ContentType = ContentType.HTML) -> ParsedDocument:
    return ParsedDocument(
        text=text, content_type=content_type, title="Source", url="https://ex.test/a"
    )


def test_parse_chunk_strategy_rejects_unknown() -> None:
    with pytest.raises(ValidationFailedError):
        parse_chunk_strategy("tokens")


def test_chunk_config_rejects_overlap_gte_size() -> None:
    with pytest.raises(ValidationFailedError):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


def test_recursive_chunker_splits_and_overlaps() -> None:
    paragraph = "Alpha sentence about transformers. " * 20
    text = f"{paragraph}\n\n{paragraph}"
    config = ChunkConfig(chunk_size=180, chunk_overlap=40, strategy=ChunkStrategy.RECURSIVE)
    chunks = RecursiveCharacterChunker(config).chunk(_doc(text))

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 180 for chunk in chunks)
    assert all(chunk.strategy is ChunkStrategy.RECURSIVE for chunk in chunks)
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    assert chunks[1].start < chunks[0].end
    assert text[chunks[0].start : chunks[0].end] == chunks[0].text
    assert chunks[0].title == "Source"
    assert chunks[0].metadata["total_chunks"] == len(chunks)


def test_recursive_chunker_keeps_short_text() -> None:
    chunks = RecursiveCharacterChunker(ChunkConfig(chunk_size=200)).chunk(
        _doc("Just a short note.")
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Just a short note."


def test_recursive_chunker_empty_returns_no_chunks() -> None:
    assert RecursiveCharacterChunker().chunk(_doc("   ")) == []


def test_recursive_chunker_duplicate_paragraphs_get_distinct_spans() -> None:
    """Repeated piece text must not reuse the first occurrence's offsets."""

    repeated = "Same paragraph content here."
    text = f"{repeated}\n\n{repeated}\n\nDifferent ending material that is unique."
    chunks = RecursiveCharacterChunker(ChunkConfig(chunk_size=40, chunk_overlap=0)).chunk(
        _doc(text)
    )

    duplicates = [chunk for chunk in chunks if chunk.text == repeated]
    assert len(duplicates) == 2
    assert duplicates[0].start == 0
    assert duplicates[1].start == text.index(repeated, duplicates[0].end)
    assert duplicates[0].start != duplicates[1].start
    for chunk in duplicates:
        assert text[chunk.start : chunk.end] == chunk.text


def test_markdown_chunker_keeps_heading_sections() -> None:
    text = """# Intro

Opening paragraph about the project.

## Setup

Install the package and configure keys.

## Usage

Run the research graph on a query.
"""
    chunks = MarkdownAwareChunker(ChunkConfig(chunk_size=400)).chunk(
        _doc(text, ContentType.MARKDOWN)
    )

    headings = [chunk.heading for chunk in chunks]
    assert "Intro" in headings
    assert "Intro > Setup" in headings
    assert "Intro > Usage" in headings
    setup = next(chunk for chunk in chunks if chunk.heading == "Intro > Setup")
    assert "Install the package" in setup.text
    assert setup.strategy is ChunkStrategy.MARKDOWN


def test_markdown_chunker_ignores_headings_inside_fences() -> None:
    text = """# Real

```python
# not a heading
print("ok")
```

Body continues after the fence.
"""
    chunks = MarkdownAwareChunker(ChunkConfig(chunk_size=500)).chunk(
        _doc(text, ContentType.MARKDOWN)
    )
    assert all(chunk.heading == "Real" or chunk.heading is None for chunk in chunks)
    assert any("# not a heading" in chunk.text for chunk in chunks)


def test_markdown_chunker_splits_oversized_section() -> None:
    body = "This is a long markdown body sentence. " * 40
    text = f"# Overflow\n\n{body}"
    chunks = MarkdownAwareChunker(ChunkConfig(chunk_size=120, chunk_overlap=20)).chunk(
        _doc(text, ContentType.MARKDOWN)
    )
    assert len(chunks) >= 2
    assert all(chunk.heading == "Overflow" for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_semantic_lexical_splits_on_topic_shift() -> None:
    text = (
        "Cats chase yarn and sleep in sunbeams. Cats purr when they are content. "
        "A cat often grooms itself after meals. "
        "Quantum entanglement links distant particles. Hilbert space describes quantum states. "
        "Measurement collapses a quantum wavefunction instantly."
    )
    chunks = SemanticChunker(
        ChunkConfig(chunk_size=400, chunk_overlap=20, semantic_threshold=0.12)
    ).chunk(_doc(text))

    assert len(chunks) >= 2
    joined_first = chunks[0].text.lower()
    joined_last = chunks[-1].text.lower()
    assert "cat" in joined_first
    assert "quantum" in joined_last
    assert chunks[0].metadata["embedder"] == "lexical"


def test_semantic_chunker_uses_injected_embeddings() -> None:
    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for item in texts:
            lower = item.lower()
            if "alpha" in lower:
                vectors.append([1.0, 0.0])
            elif "beta" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors

    text = "Alpha one. Alpha two. Alpha three. Beta one. Beta two. Beta three."
    chunks = SemanticChunker(
        ChunkConfig(chunk_size=80, chunk_overlap=0, semantic_threshold=0.5),
        embed_text=embed,
    ).chunk(_doc(text))

    assert len(chunks) >= 2
    assert "alpha" in chunks[0].text.lower()
    assert "beta" in chunks[-1].text.lower()
    assert chunks[0].metadata["embedder"] == "custom"


def test_auto_strategy_uses_markdown_for_markdown_documents() -> None:
    text = "# Title\n\nA short body paragraph."
    chunks = chunk_document(
        _doc(text, ContentType.MARKDOWN),
        config=ChunkConfig(chunk_size=200, strategy=ChunkStrategy.AUTO),
    )
    assert chunks[0].strategy is ChunkStrategy.MARKDOWN
    assert chunks[0].heading == "Title"


def test_auto_strategy_uses_recursive_for_html() -> None:
    chunks = chunk_document(
        _doc("A short HTML article about research methods."),
        config=ChunkConfig(chunk_size=200, strategy=ChunkStrategy.AUTO),
    )
    assert len(chunks) == 1
    assert chunks[0].strategy is ChunkStrategy.RECURSIVE


def test_service_parse_and_chunk() -> None:
    service = DocumentParseService(
        build_parsers(ocr_extract_text=lambda _: "x"),
        chunk_config=ChunkConfig(chunk_size=80, chunk_overlap=10),
    )
    chunks = service.parse_and_chunk(
        "# Hello\n\nUseful content about LangGraph agents and research workflows.\n\n"
        "## More\n\nAnother section with additional useful detail for chunking tests.",
        filename="readme.md",
        url="https://example.com/readme.md",
    )

    assert chunks
    assert all(isinstance(chunk.index, int) for chunk in chunks)
    assert chunks[0].content_type is ContentType.MARKDOWN
    assert chunks[0].url == "https://example.com/readme.md"
    assert any(chunk.heading for chunk in chunks)


def test_service_parse_and_chunk_many_soft_fails() -> None:
    from insightforge.infrastructure.document import ParseRequest

    service = DocumentParseService(
        build_parsers(ocr_extract_text=lambda _: "x"),
        chunk_config=ChunkConfig(chunk_size=120, chunk_overlap=10),
    )
    chunks = service.parse_and_chunk_many(
        [
            ParseRequest(raw="# Good\n\nEnough words to keep.", filename="a.md"),
            ParseRequest(raw="{bad", filename="b.json"),
            ParseRequest(raw='{"content": "fine payload text here"}', filename="c.json"),
        ]
    )
    assert len(chunks) >= 2
    assert any(chunk.content_type is ContentType.MARKDOWN for chunk in chunks)
    assert any(chunk.content_type is ContentType.JSON for chunk in chunks)
