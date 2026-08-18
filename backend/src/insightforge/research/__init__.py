"""Research pipeline features shared by reasoning, reflection, and report agents.

Each module is a pure-function toolkit (no I/O, no LLM). Agents compose them
so the deterministic and future LLM-backed implementations produce the same
domain outputs.
"""

from insightforge.research.aggregation import aggregate_evidence
from insightforge.research.citations import (
    build_citations,
    link_clusters_to_citations,
)
from insightforge.research.compression import (
    compress_documents,
    compress_hits,
)
from insightforge.research.confidence import (
    reasoning_confidence,
    reflection_confidence,
)
from insightforge.research.conflicts import detect_conflicts

__all__ = [
    "aggregate_evidence",
    "build_citations",
    "compress_documents",
    "compress_hits",
    "detect_conflicts",
    "link_clusters_to_citations",
    "reasoning_confidence",
    "reflection_confidence",
]
