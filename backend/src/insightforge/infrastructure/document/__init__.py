"""Document processing infrastructure — parsers, cleaning, chunking, citation."""

from insightforge.infrastructure.document.base import DocumentParser
from insightforge.infrastructure.document.chunking import (
    ChunkConfig,
    DocumentChunker,
    EmbedFn,
    MarkdownAwareChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    chunk_document,
    parse_chunk_strategy,
)
from insightforge.infrastructure.document.citation import (
    build_citation,
    cite_chunks,
    cite_document,
    format_citation,
    normalize_date,
    parse_authors,
)
from insightforge.infrastructure.document.cleaning import (
    clean_document,
    extract_text_metadata,
    normalize_text,
    remove_boilerplate,
)
from insightforge.infrastructure.document.detect import detect_content_type
from insightforge.infrastructure.document.parsers import (
    HtmlDocumentParser,
    JsonDocumentParser,
    MarkdownDocumentParser,
    OcrDocumentParser,
    PdfDocumentParser,
)
from insightforge.infrastructure.document.service import (
    DocumentParseService,
    ParseRequest,
    build_parsers,
    create_document_parse_service,
)

__all__ = [
    "ChunkConfig",
    "DocumentChunker",
    "DocumentParseService",
    "DocumentParser",
    "EmbedFn",
    "HtmlDocumentParser",
    "JsonDocumentParser",
    "MarkdownAwareChunker",
    "MarkdownDocumentParser",
    "OcrDocumentParser",
    "ParseRequest",
    "PdfDocumentParser",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "build_citation",
    "build_parsers",
    "chunk_document",
    "cite_chunks",
    "cite_document",
    "clean_document",
    "create_document_parse_service",
    "detect_content_type",
    "extract_text_metadata",
    "format_citation",
    "normalize_date",
    "normalize_text",
    "parse_authors",
    "parse_chunk_strategy",
    "remove_boilerplate",
]
