"""Content-type detection from MIME, filename, URL, and magic bytes."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from insightforge.core.logging import get_logger
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)

_MIME_MAP: dict[str, ContentType] = {
    "text/html": ContentType.HTML,
    "application/xhtml+xml": ContentType.HTML,
    "text/markdown": ContentType.MARKDOWN,
    "text/x-markdown": ContentType.MARKDOWN,
    "application/pdf": ContentType.PDF,
    "application/json": ContentType.JSON,
    "text/json": ContentType.JSON,
    "image/png": ContentType.OCR,
    "image/jpeg": ContentType.OCR,
    "image/jpg": ContentType.OCR,
    "image/tiff": ContentType.OCR,
    "image/webp": ContentType.OCR,
    "image/bmp": ContentType.OCR,
    "image/gif": ContentType.OCR,
}

_EXT_MAP: dict[str, ContentType] = {
    ".html": ContentType.HTML,
    ".htm": ContentType.HTML,
    ".xhtml": ContentType.HTML,
    ".md": ContentType.MARKDOWN,
    ".markdown": ContentType.MARKDOWN,
    ".pdf": ContentType.PDF,
    ".json": ContentType.JSON,
    ".png": ContentType.OCR,
    ".jpg": ContentType.OCR,
    ".jpeg": ContentType.OCR,
    ".tif": ContentType.OCR,
    ".tiff": ContentType.OCR,
    ".webp": ContentType.OCR,
    ".bmp": ContentType.OCR,
    ".gif": ContentType.OCR,
}

_HTML_HINT = re.compile(rb"<!doctype\s+html|<html[\s>]|</html>", re.IGNORECASE)
_PDF_MAGIC = b"%PDF"
_IMAGE_MAGICS: tuple[tuple[bytes, ContentType], ...] = (
    (b"\x89PNG\r\n\x1a\n", ContentType.OCR),
    (b"\xff\xd8\xff", ContentType.OCR),
    (b"II*\x00", ContentType.OCR),
    (b"MM\x00*", ContentType.OCR),
    (b"RIFF", ContentType.OCR),  # webp starts with RIFF....WEBP
    (b"BM", ContentType.OCR),
    (b"GIF87a", ContentType.OCR),
    (b"GIF89a", ContentType.OCR),
)


def _normalize_mime(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower() or None


def _extension_from(filename: str | None = None, url: str | None = None) -> str | None:
    candidate = filename
    if not candidate and url:
        path = urlparse(url).path
        candidate = path.rsplit("/", 1)[-1] if path else None
    if not candidate:
        return None
    suffix = Path(candidate).suffix.lower()
    return suffix or None


def _sniff_raw(raw: bytes | str | None) -> ContentType | None:
    if raw is None:
        return None

    if isinstance(raw, str):
        sample = raw.lstrip()
        if not sample:
            return None
        if sample[0] in "{[":
            return ContentType.JSON
        if sample.startswith("---") or sample.startswith("#"):
            return ContentType.MARKDOWN
        lower = sample[:500].lower()
        if "<!doctype html" in lower or "<html" in lower:
            return ContentType.HTML
        return None

    if not raw:
        return None
    if raw.startswith(_PDF_MAGIC):
        return ContentType.PDF
    for magic, content_type in _IMAGE_MAGICS:
        if raw.startswith(magic):
            # WebP: RIFF....WEBP
            if magic == b"RIFF" and (len(raw) < 12 or raw[8:12] != b"WEBP"):
                continue
            return content_type
    head = raw[:2048]
    if _HTML_HINT.search(head):
        return ContentType.HTML
    stripped = raw.lstrip()
    if stripped[:1] in (b"{", b"["):
        return ContentType.JSON
    return None


def detect_content_type(
    *,
    content_type: str | None = None,
    filename: str | None = None,
    url: str | None = None,
    raw: bytes | str | None = None,
) -> ContentType | None:
    """Best-effort content type from MIME, path, URL, then payload sniffing."""

    mime = _normalize_mime(content_type)
    if mime and mime in _MIME_MAP:
        detected = _MIME_MAP[mime]
        logger.debug("content type from mime=%s -> %s", mime, detected.value)
        return detected

    # Explicit enum / short name (e.g. "markdown", "pdf")
    if mime:
        try:
            detected = ContentType(mime)
            logger.debug("content type from name=%s", detected.value)
            return detected
        except ValueError:
            pass

    ext = _extension_from(filename, url)
    if ext and ext in _EXT_MAP:
        detected = _EXT_MAP[ext]
        logger.debug("content type from ext=%s -> %s", ext, detected.value)
        return detected

    sniffed = _sniff_raw(raw)
    if sniffed is not None:
        logger.debug("content type from sniff -> %s", sniffed.value)
        return sniffed

    logger.debug(
        "content type unresolved mime=%s filename=%s url=%s",
        mime,
        filename,
        url,
    )
    return None
