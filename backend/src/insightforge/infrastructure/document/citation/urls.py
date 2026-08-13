"""URL helpers for citations."""

from __future__ import annotations

from urllib.parse import urlparse


def site_name(url: str | None) -> str | None:
    """Return a display hostname (``example.com``) from ``url``."""

    if not url or not url.strip():
        return None
    host = (urlparse(url.strip()).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def clean_url(url: str | None) -> str | None:
    """Trim a source URL for citation display. Empty strings become ``None``."""

    if url is None:
        return None
    cleaned = url.strip()
    return cleaned or None
