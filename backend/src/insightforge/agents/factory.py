"""Choose heuristic or Gemini-backed agents from an optional LLM service."""

from __future__ import annotations

from insightforge.agents.planner.base import Planner
from insightforge.agents.planner.llm import LlmPlanner
from insightforge.agents.planner.simple import SimplePlanner
from insightforge.agents.reasoner.base import Reasoner
from insightforge.agents.reasoner.llm import LlmReasoner
from insightforge.agents.reasoner.simple import SimpleReasoner
from insightforge.agents.reflection import (
    LlmReflectionAgent,
    ReflectionAgent,
    SimpleReflectionAgent,
)
from insightforge.agents.report.base import ReportGenerator
from insightforge.agents.report.llm import LlmReportGenerator
from insightforge.agents.report.simple import SimpleReportGenerator
from insightforge.infrastructure.llm import LlmService


def create_planner(llm: LlmService | None = None) -> Planner:
    """Return an LLM planner when Gemini is available."""

    if llm is not None and llm.available:
        return LlmPlanner(llm)
    return SimplePlanner()


def create_reasoner(llm: LlmService | None = None) -> Reasoner:
    """Return an LLM reasoner when Gemini is available."""

    if llm is not None and llm.available:
        return LlmReasoner(llm)
    return SimpleReasoner()


def create_report_generator(llm: LlmService | None = None) -> ReportGenerator:
    """Return an LLM report generator when Gemini is available."""

    if llm is not None and llm.available:
        return LlmReportGenerator(llm)
    return SimpleReportGenerator()


def create_reflection_agent(llm: LlmService | None = None) -> ReflectionAgent:
    """Return an LLM reflection agent when Gemini is available."""

    if llm is not None and llm.available:
        return LlmReflectionAgent(llm)
    return SimpleReflectionAgent()
