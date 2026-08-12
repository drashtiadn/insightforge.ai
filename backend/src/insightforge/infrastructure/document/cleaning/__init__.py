"""Phase 4.2 document cleaning — boilerplate, normalize, metadata."""

from __future__ import annotations

from insightforge.core.logging import get_logger
from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document.cleaning.boilerplate import remove_boilerplate
from insightforge.infrastructure.document.cleaning.metadata import extract_text_metadata
from insightforge.infrastructure.document.cleaning.normalize import normalize_text

logger = get_logger(__name__)

__all__ = [
    "clean_document",
    "extract_text_metadata",
    "normalize_text",
    "remove_boilerplate",
]


def clean_document(
    document: ParsedDocument,
    *,
    remove_chrome: bool = True,
    normalize: bool = True,
    extract_metadata: bool = True,
) -> ParsedDocument:
    """Clean a parsed document in place-pipeline order.

    Order: boilerplate removal → text normalization → metadata extraction.
    """

    before = len(document.text)
    text = document.text

    if remove_chrome:
        text = remove_boilerplate(text, content_type=document.content_type)
    if normalize:
        text = normalize_text(text)

    metadata = dict(document.metadata)
    if extract_metadata:
        metadata = extract_text_metadata(
            text,
            existing=metadata,
            title=document.title,
            url=document.url,
        )
    else:
        metadata["char_count"] = len(text)
        metadata["cleaned"] = True

    metadata["chars_before_clean"] = before
    metadata["chars_after_clean"] = len(text)

    logger.info(
        "document cleaned content_type=%s chars_before=%d chars_after=%d",
        document.content_type.value,
        before,
        len(text),
        extra={
            "content_type": document.content_type.value,
            "chars_before": before,
            "chars_after": len(text),
        },
    )
    return document.model_copy(update={"text": text, "metadata": metadata})
