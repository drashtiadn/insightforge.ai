"""Normalize citation dates to ISO ``YYYY-MM-DD`` when possible."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_PDF_RE = re.compile(r"^D:(\d{4})(\d{2})(\d{2})")
_MDY_RE = re.compile(
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})",
    re.IGNORECASE,
)
_DMY_RE = re.compile(
    r"(\d{1,2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+(\d{4})",
    re.IGNORECASE,
)


def normalize_date(value: Any) -> str | None:
    """Return ISO date when parseable, otherwise the original non-empty string."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()
    if not text:
        return None

    pdf = _PDF_RE.match(text)
    if pdf:
        return _safe_iso(pdf.group(1), pdf.group(2), pdf.group(3)) or text

    iso = _ISO_RE.search(text)
    if iso:
        return _safe_iso(iso.group(1), iso.group(2), iso.group(3)) or text

    mdy = _MDY_RE.search(text)
    if mdy:
        month = _MONTHS[mdy.group(1).lower()]
        return _safe_iso(mdy.group(3), month, mdy.group(2)) or text

    dmy = _DMY_RE.search(text)
    if dmy:
        month = _MONTHS[dmy.group(2).lower()]
        return _safe_iso(dmy.group(3), month, dmy.group(1)) or text

    return text


def _safe_iso(year: str | int, month: str | int, day: str | int) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None
