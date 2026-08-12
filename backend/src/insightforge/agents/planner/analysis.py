"""Query analysis and intent detection for the planner."""

from __future__ import annotations

import re

from insightforge.agents.planner.schemas import QueryAnalysis
from insightforge.core.logging import get_logger
from insightforge.prompts.planner import INTENT_PATTERNS, STOPWORDS
from insightforge.shared.enums import QueryIntent

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Strip and collapse whitespace without changing meaning."""

    return _WHITESPACE_RE.sub(" ", query.strip())


def extract_keywords(normalized_query: str) -> list[str]:
    """Pull content tokens, dropping common stopwords."""

    tokens = _TOKEN_RE.findall(normalized_query.lower())
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def detect_intent(normalized_query: str) -> QueryIntent:
    """Classify intent with ordered keyword heuristics.

    Falls back to ``exploratory`` when no pattern matches — safe default for
    broad research queries.
    """

    padded = f" {normalized_query.lower()} "
    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if pattern in padded:
                logger.debug(
                    "intent matched pattern=%r intent=%s",
                    pattern.strip(),
                    intent.value,
                )
                return intent
    return QueryIntent.EXPLORATORY


def analyze_query(query: str) -> QueryAnalysis:
    """Normalize the query and extract keywords for decomposition."""

    normalized = normalize_query(query)
    keywords = extract_keywords(normalized)
    tokens = _TOKEN_RE.findall(normalized.lower())
    analysis = QueryAnalysis(
        original_query=query,
        normalized_query=normalized,
        keywords=keywords,
        token_count=len(tokens),
    )
    logger.debug(
        "query analyzed token_count=%d keyword_count=%d",
        analysis.token_count,
        len(analysis.keywords),
    )
    return analysis
