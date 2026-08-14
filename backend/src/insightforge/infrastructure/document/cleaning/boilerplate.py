"""Remove common chrome / boilerplate lines from extracted document text."""

from __future__ import annotations

import re

from insightforge.core.logging import get_logger
from insightforge.shared.enums import ContentType

logger = get_logger(__name__)

# Whole-line patterns typical of web chrome, footers, and share widgets.
_BOILERPLATE_LINE = re.compile(
    r"""(?ix)^\s*(?:
        (?:skip\s+to\s+(?:main\s+)?content)
      | (?:cookie|privacy)\s+(?:policy|notice|settings|consent|preferences)
      | (?:we\s+use\s+cookies|accept\s+(?:all\s+)?cookies|manage\s+cookies)
      | (?:all\s+rights\s+reserved|©|\(c\)\s*\d{4})
      | (?:subscribe(?:\s+to\s+(?:our\s+)?newsletter)?)
      | (?:sign\s+up|log\s+in|sign\s+in|create\s+(?:an\s+)?account)
      | (?:share\s+(?:this|on)|tweet\s+this|follow\s+us)
      | (?:related\s+(?:articles?|posts?|stories)|you\s+may\s+also\s+like)
      | (?:advertisement|sponsored\s+content|advertising)
      | (?:terms\s+of\s+(?:use|service)|disclaimer)
      | (?:back\s+to\s+top|table\s+of\s+contents)
      | (?:menu|navigation|breadcrumb)
    )\s*$"""
)

_MARKDOWN_ONLY_LINK = re.compile(r"^\[([^\]]+)\]\([^)]+\)\.?$")
_SHARE_LABEL = re.compile(r"(?i)^(share|tweet|facebook|linkedin|reddit|email|print|copy\s+link)$")


def remove_boilerplate(text: str, *, content_type: ContentType | None = None) -> str:
    """Drop boilerplate lines while preserving substantive content."""

    if not text.strip():
        return ""

    kept: list[str] = []
    removed = 0
    previous: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            # Keep blank lines as separators; normalize later collapses runs.
            if kept and kept[-1] != "":
                kept.append("")
            continue

        if _is_boilerplate_line(stripped, content_type=content_type):
            removed += 1
            continue

        # Collapse consecutive duplicate lines (common in PDF/OCR chrome).
        if previous is not None and stripped == previous:
            removed += 1
            continue

        kept.append(stripped)
        previous = stripped

    cleaned = "\n".join(kept).strip()
    logger.debug(
        "remove_boilerplate content_type=%s removed_lines=%d chars_after=%d",
        content_type.value if content_type else None,
        removed,
        len(cleaned),
    )
    return cleaned


def _is_boilerplate_line(line: str, *, content_type: ContentType | None) -> bool:
    if _BOILERPLATE_LINE.match(line):
        return True

    # Lone markdown share links, e.g. [Twitter](https://...)
    if content_type is ContentType.MARKDOWN or content_type is None:
        link = _MARKDOWN_ONLY_LINK.match(line)
        if link and _SHARE_LABEL.match(link.group(1).strip()):
            return True

    # Extremely short chrome tokens that are not markdown headings.
    if len(line) <= 2 and not line.startswith("#"):
        return line in {"|", "-", "\u2022", "\u00b7", "\u203a", "\u00bb", "\u00ab"}

    return False
