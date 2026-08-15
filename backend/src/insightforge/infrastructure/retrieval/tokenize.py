"""Tokenization helpers for BM25 retrieval."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercased word-like tokens, preserving duplicates for term frequency."""

    return [token.lower() for token in _TOKEN_RE.findall(text)]
