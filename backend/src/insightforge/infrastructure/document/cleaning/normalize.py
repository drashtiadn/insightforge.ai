"""Unicode and whitespace normalization for parsed document text."""

from __future__ import annotations

import re
import unicodedata

from insightforge.core.logging import get_logger

logger = get_logger(__name__)

# Soft hyphen, zero-width, and BOM-like noise often left by PDF/OCR/HTML.
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff\u2060"), None)
_MULTI_SPACE = re.compile(r"[^\S\n]+")
_MULTI_BLANK = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize Unicode, strip invisible chars, and tidy whitespace."""

    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.translate(_INVISIBLE)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = _MULTI_BLANK.sub("\n\n", cleaned).strip()

    logger.debug(
        "normalize_text chars_before=%d chars_after=%d",
        len(text),
        len(cleaned),
    )
    return cleaned
