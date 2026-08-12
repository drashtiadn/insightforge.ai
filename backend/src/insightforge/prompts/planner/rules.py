"""Heuristic rules for query analysis and intent detection.

Kept here so an LLM-backed planner can later reuse the same vocabulary
without changing agent code.
"""

from __future__ import annotations

from insightforge.shared.enums import QueryIntent

# Checked in order — first match wins.
INTENT_PATTERNS: tuple[tuple[QueryIntent, tuple[str, ...]], ...] = (
    (
        QueryIntent.COMPARATIVE,
        (
            " vs ",
            " versus ",
            "compare ",
            "comparison",
            "difference between",
            "differences between",
            "better than",
            "pros and cons",
        ),
    ),
    (
        QueryIntent.HOWTO,
        (
            "how to ",
            "how do i ",
            "how can i ",
            "tutorial",
            "step by step",
            "walkthrough",
            "guide to ",
        ),
    ),
    (
        QueryIntent.NEWS,
        (
            "latest ",
            "recent ",
            "breaking ",
            "today ",
            "this week",
            "this month",
            "current events",
            " news",
        ),
    ),
    (
        QueryIntent.ACADEMIC,
        (
            "arxiv",
            "peer-reviewed",
            "peer reviewed",
            "journal",
            "doi",
            "whitepaper",
            "white paper",
            "literature review",
            "research paper",
            "scientific study",
        ),
    ),
    (
        QueryIntent.CODE,
        (
            "github",
            "repository",
            " source code",
            "npm ",
            "pypi",
            "sdk",
            "api reference",
            "implement ",
            "library for",
            "package for",
        ),
    ),
    (
        QueryIntent.FACTUAL,
        (
            "what is ",
            "what are ",
            "who is ",
            "who was ",
            "when did ",
            "where is ",
            "define ",
            "definition of ",
            "meaning of ",
        ),
    ),
)

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "into",
        "about",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "how",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "we",
        "they",
        "me",
        "my",
        "your",
        "our",
        "please",
        "tell",
        "find",
        "show",
        "give",
        "get",
    }
)

__all__ = [
    "INTENT_PATTERNS",
    "STOPWORDS",
]
