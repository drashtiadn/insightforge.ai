"""Document parse service — detect format, select parser, extract text."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document, ParsedDocument
from insightforge.infrastructure.document.base import DocumentParser
from insightforge.infrastructure.document.cleaning import clean_document
from insightforge.infrastructure.document.detect import detect_content_type
from insightforge.infrastructure.document.parsers import (
    HtmlDocumentParser,
    JsonDocumentParser,
    MarkdownDocumentParser,
    OcrDocumentParser,
    PdfDocumentParser,
)
from insightforge.infrastructure.document.parsers.ocr import OcrExtractFn
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)


class ParseRequest(BaseModel):
    """One raw payload to parse (used by ``parse_many``)."""

    raw: bytes | str
    content_type: str | None = None
    filename: str | None = None
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_parsers(
    *,
    tesseract_cmd: str | None = None,
    ocr_extract_text: OcrExtractFn | None = None,
) -> list[DocumentParser]:
    """Construct the default parser set for Phase 4.1."""

    return [
        HtmlDocumentParser(),
        MarkdownDocumentParser(),
        PdfDocumentParser(),
        JsonDocumentParser(),
        OcrDocumentParser(
            extract_text=ocr_extract_text,
            tesseract_cmd=tesseract_cmd,
        ),
    ]


class DocumentParseService:
    """Route raw documents to the matching format parser.

    Pipeline: detect → parse → optional clean (Phase 4.2).
    Unknown formats and unavailable parsers raise ``ValidationFailedError`` for a
    single ``parse`` call. ``parse_many`` soft-fails per item so one bad source
    does not abort the batch.
    """

    def __init__(
        self,
        parsers: Sequence[DocumentParser],
        *,
        cleaning: bool = True,
    ) -> None:
        self._parsers = list(parsers)
        self._cleaning = cleaning

    @property
    def parsers(self) -> list[DocumentParser]:
        return list(self._parsers)

    @property
    def cleaning(self) -> bool:
        return self._cleaning

    def get_parser(self, content_type: ContentType) -> DocumentParser | None:
        for parser in self._parsers:
            if parser.name is content_type:
                return parser
        return None

    def resolve_parser(
        self,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        url: str | None = None,
        raw: bytes | str | None = None,
    ) -> DocumentParser:
        """Pick a registered parser from MIME / filename / URL / payload hints."""

        detected = detect_content_type(
            content_type=content_type,
            filename=filename,
            url=url,
            raw=raw,
        )
        if detected is None:
            raise ValidationFailedError(
                "unable to detect document content type",
                details={
                    "content_type": content_type,
                    "filename": filename,
                    "url": url,
                },
            )

        parser = self.get_parser(detected)
        if parser is None:
            raise ValidationFailedError(
                f"no parser registered for content type {detected.value}",
                details={"content_type": detected.value},
            )
        if not parser.available:
            raise ValidationFailedError(
                f"{detected.value} parser unavailable",
                details={"content_type": detected.value},
            )
        return parser

    def parse(
        self,
        raw: bytes | str,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        url: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        cleaning: bool | None = None,
    ) -> ParsedDocument:
        """Parse a single raw payload into a ``ParsedDocument``."""

        size = len(raw) if isinstance(raw, bytes | str) else -1
        logger.info(
            "document parse started size=%d content_type=%s filename=%s",
            size,
            content_type,
            filename,
        )
        parser = self.resolve_parser(
            content_type=content_type,
            filename=filename,
            url=url,
            raw=raw,
        )
        parsed = parser.parse(
            raw,
            source_url=url,
            filename=filename,
            title=title,
            extra_metadata=metadata,
        )
        apply_cleaning = self._cleaning if cleaning is None else cleaning
        if apply_cleaning:
            parsed = clean_document(parsed)
        logger.info(
            "document parse finished parser=%s chars=%d cleaned=%s",
            parser.name.value,
            len(parsed.text),
            apply_cleaning,
            extra={"parser": parser.name.value, "char_count": len(parsed.text)},
        )
        return parsed

    def parse_source(
        self,
        document: Document,
        *,
        raw: bytes | str | None = None,
        cleaning: bool | None = None,
    ) -> ParsedDocument:
        """Parse payload associated with a search ``Document``."""

        payload: bytes | str
        if raw is not None:
            payload = raw
        elif document.content:
            payload = document.content
        else:
            raise ValidationFailedError(
                "document has no content to parse",
                details={"url": document.url, "title": document.title},
            )

        meta = dict(document.metadata)
        meta.setdefault("provider", document.provider.value)
        return self.parse(
            payload,
            content_type=_as_optional_str(meta.get("content_type")),
            filename=_as_optional_str(meta.get("filename")),
            url=document.url,
            title=document.title,
            metadata=meta,
            cleaning=cleaning,
        )

    def parse_many(self, requests: Sequence[ParseRequest]) -> list[ParsedDocument]:
        """Parse many payloads; skip items that fail validation or parsing."""

        logger.info("document parse_many started count=%d", len(requests))
        results: list[ParsedDocument] = []
        for index, request in enumerate(requests):
            try:
                results.append(
                    self.parse(
                        request.raw,
                        content_type=request.content_type,
                        filename=request.filename,
                        url=request.url,
                        title=request.title,
                        metadata=request.metadata or None,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "document parse_many item failed index=%d error=%s",
                    index,
                    exc,
                    extra={"index": index},
                )
                continue
        logger.info(
            "document parse_many finished ok=%d failed=%d",
            len(results),
            len(requests) - len(results),
        )
        return results


def create_document_parse_service(
    settings: Settings | None = None,
    *,
    ocr_extract_text: OcrExtractFn | None = None,
) -> DocumentParseService:
    """Factory used by application code and tests."""

    cfg = settings or get_settings()
    return DocumentParseService(
        build_parsers(
            tesseract_cmd=cfg.tesseract_cmd,
            ocr_extract_text=ocr_extract_text,
        ),
        cleaning=cfg.document_cleaning_enabled,
    )


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
