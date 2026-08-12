"""Document parser contract shared by every content format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from insightforge.domain.models import ParsedDocument
from insightforge.shared.enums import ContentType


class DocumentParser(ABC):
    """One content-format parser (HTML, Markdown, PDF, OCR, JSON).

    Parsers must raise ``ValidationFailedError`` for empty or unusable input.
    Corrupt or unreadable payloads should raise ``ValidationFailedError`` so the
    service can soft-fail per item without treating it as an upstream outage.
    """

    name: ContentType

    @property
    def available(self) -> bool:
        """True when required runtime deps are present and the parser can run."""

        return True

    def supports(
        self,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        url: str | None = None,
        raw: bytes | str | None = None,
    ) -> bool:
        """Return True when this parser should handle the given hints."""

        from insightforge.infrastructure.document.detect import detect_content_type

        detected = detect_content_type(
            content_type=content_type,
            filename=filename,
            url=url,
            raw=raw,
        )
        return detected is self.name

    @abstractmethod
    def parse(
        self,
        raw: bytes | str,
        *,
        source_url: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """Extract normalized text and light metadata from ``raw``."""
