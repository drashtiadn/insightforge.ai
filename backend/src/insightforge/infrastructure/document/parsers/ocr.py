"""OCR document parser for image payloads."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document.base import DocumentParser
from insightforge.infrastructure.document.helpers import ensure_bytes, merge_metadata
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)

OcrExtractFn = Callable[[bytes], str]


def _tesseract_available(cmd: str | None = None) -> bool:
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError:
        return False

    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        pytesseract.get_tesseract_version()
    except (TesseractNotFoundError, OSError):
        return False
    except Exception:
        return False
    return True


def default_tesseract_extract(image_bytes: bytes, *, cmd: str | None = None) -> str:
    """Run Tesseract OCR on image bytes."""

    import pytesseract
    from PIL import Image

    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    with Image.open(BytesIO(image_bytes)) as image:
        # Force load so file handle can close cleanly.
        image.load()
        rgb = image.convert("RGB")
        return str(pytesseract.image_to_string(rgb))


class OcrDocumentParser(DocumentParser):
    """Extract text from images via Tesseract (or an injected extract function).

    Inject ``extract_text`` in tests to avoid requiring a system Tesseract binary.
    """

    name = ContentType.OCR

    def __init__(
        self,
        *,
        extract_text: OcrExtractFn | None = None,
        tesseract_cmd: str | None = None,
    ) -> None:
        self._extract_text = extract_text
        self._tesseract_cmd = tesseract_cmd

    @property
    def available(self) -> bool:
        if self._extract_text is not None:
            return True
        return _tesseract_available(self._tesseract_cmd)

    def parse(
        self,
        raw: bytes | str,
        *,
        source_url: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        if not self.available:
            raise ValidationFailedError(
                "OCR parser unavailable (Tesseract not configured)",
                details={"parser": self.name.value},
            )

        payload = ensure_bytes(raw)
        logger.info("ocr parse started bytes=%d", len(payload))

        try:
            if self._extract_text is not None:
                text = self._extract_text(payload)
            else:
                text = default_tesseract_extract(payload, cmd=self._tesseract_cmd)
        except ValidationFailedError:
            raise
        except Exception as exc:
            logger.warning("ocr parse failed error=%s", exc)
            raise ValidationFailedError(
                "unable to OCR image payload",
                details={"parser": self.name.value, "error": str(exc)},
            ) from exc

        cleaned = (text or "").strip()
        metadata = merge_metadata(
            parser=self.name.value,
            filename=filename,
            extra=extra_metadata,
            extracted={
                "engine": "custom" if self._extract_text is not None else "tesseract",
                "char_count": len(cleaned),
            },
        )
        logger.info("ocr parse finished chars=%d", len(cleaned))
        return ParsedDocument(
            text=cleaned,
            content_type=self.name,
            title=title,
            url=source_url,
            metadata=metadata,
        )
