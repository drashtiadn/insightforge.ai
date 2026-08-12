"""Document processing infrastructure — parsers that normalize research sources."""

from insightforge.infrastructure.document.base import DocumentParser
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
    "DocumentParseService",
    "DocumentParser",
    "HtmlDocumentParser",
    "JsonDocumentParser",
    "MarkdownDocumentParser",
    "OcrDocumentParser",
    "ParseRequest",
    "PdfDocumentParser",
    "build_parsers",
    "create_document_parse_service",
    "detect_content_type",
]
