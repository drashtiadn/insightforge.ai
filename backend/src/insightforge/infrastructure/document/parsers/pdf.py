"""PDF document parser."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from pypdf import PdfReader

from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document.base import DocumentParser
from insightforge.infrastructure.document.helpers import ensure_bytes, merge_metadata
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)


def _meta_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class PdfDocumentParser(DocumentParser):
    """Extract text and document info from PDF bytes."""

    name = ContentType.PDF

    def parse(
        self,
        raw: bytes | str,
        *,
        source_url: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        payload = ensure_bytes(raw)
        logger.info("pdf parse started bytes=%d", len(payload))

        try:
            reader = PdfReader(BytesIO(payload))
        except Exception as exc:
            logger.warning("pdf parse failed error=%s", exc)
            raise ValidationFailedError(
                "unable to read PDF payload",
                details={"parser": self.name.value, "error": str(exc)},
            ) from exc

        if getattr(reader, "is_encrypted", False):
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise ValidationFailedError(
                    "encrypted PDF cannot be opened",
                    details={"parser": self.name.value},
                ) from exc
            if unlocked == 0:
                raise ValidationFailedError(
                    "encrypted PDF cannot be opened",
                    details={"parser": self.name.value},
                )

        pages: list[str] = []
        for index, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
            except Exception as exc:
                logger.warning("pdf page extract failed page=%d error=%s", index, exc)
                extracted = ""
            if extracted.strip():
                pages.append(extracted.strip())

        text = "\n\n".join(pages).strip()
        info = reader.metadata
        pdf_title = title or (_meta_str(info.title) if info else None)
        author = _meta_str(info.author) if info else None
        created = _meta_str(getattr(info, "creation_date", None)) if info else None

        metadata = merge_metadata(
            parser=self.name.value,
            filename=filename,
            extra=extra_metadata,
            extracted={
                "page_count": len(reader.pages),
                "author": author,
                "created": created,
                "char_count": len(text),
            },
        )
        logger.info(
            "pdf parse finished pages=%d chars=%d title=%s",
            len(reader.pages),
            len(text),
            bool(pdf_title),
        )
        return ParsedDocument(
            text=text,
            content_type=self.name,
            title=pdf_title,
            url=source_url,
            metadata=metadata,
        )
