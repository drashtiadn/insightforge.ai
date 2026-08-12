"""Concrete document parser implementations."""

from insightforge.infrastructure.document.parsers.html import HtmlDocumentParser
from insightforge.infrastructure.document.parsers.json_parser import JsonDocumentParser
from insightforge.infrastructure.document.parsers.markdown import MarkdownDocumentParser
from insightforge.infrastructure.document.parsers.ocr import OcrDocumentParser
from insightforge.infrastructure.document.parsers.pdf import PdfDocumentParser

__all__ = [
    "HtmlDocumentParser",
    "JsonDocumentParser",
    "MarkdownDocumentParser",
    "OcrDocumentParser",
    "PdfDocumentParser",
]
