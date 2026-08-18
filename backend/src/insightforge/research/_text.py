"""Internal text helpers shared by the research feature modules."""

from __future__ import annotations

import re

from insightforge.prompts.planner import STOPWORDS

__all__ = [
    "clean_snippet",
    "content_tokens",
    "jaccard",
    "normalize",
]

# Word-like tokens for semantic-overlap checks. Deliberately stricter than
# the BM25 tokenizer (no dots, no dashes) so "embeddings" and "embeddings."
# collide, and so hyphenated phrases split into their meaningful parts.
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalize(text: str) -> str:
    """Collapse whitespace and strip surrounding padding."""

    return " ".join(text.split())


def clean_snippet(text: str, *, limit: int = 240) -> str:
    """Normalize and cap ``text`` for use as a display snippet or claim."""

    cleaned = normalize(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def content_tokens(text: str) -> set[str]:
    """Return a set of lowercase content tokens (stopwords removed)."""

    return {
        token.lower()
        for token in _WORD_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    }


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard similarity between two token sets; 0 for empty inputs."""

    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
