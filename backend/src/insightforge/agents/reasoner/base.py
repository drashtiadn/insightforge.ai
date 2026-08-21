"""Reasoner agent interface — synthesize an answer from ranked evidence."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from insightforge.agents.base import BaseAgent
from insightforge.agents.tools import Tool
from insightforge.domain.models import Document, ReasoningResult, RetrievalHit


class Reasoner(BaseAgent):
    """Contract for reasoning agents.

    Implementations receive a query plus supporting evidence (retrieval hits
    from the RAG layer and/or raw search documents) and return a structured
    ``ReasoningResult``. Downstream reflection and report agents consume the
    result — they never reach back into raw evidence blindly.
    """

    name = "reasoner"

    def __init__(self, tools: list[Tool] | None = None) -> None:
        super().__init__(tools=tools)

    @abstractmethod
    def reason(
        self,
        query: str,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
        feedback: str | None = None,
    ) -> ReasoningResult:
        """Return a structured reasoning result for ``query``."""

    def run(
        self,
        query: str,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
        feedback: str | None = None,
    ) -> ReasoningResult:
        """``BaseAgent`` entry point — delegates to ``reason``."""

        return self.reason(query, hits=hits, documents=documents, feedback=feedback)
