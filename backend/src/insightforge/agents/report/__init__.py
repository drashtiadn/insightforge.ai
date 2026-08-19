"""Report generator agent package."""

from insightforge.agents.report.base import ReportGenerator
from insightforge.agents.report.llm import LlmReportGenerator
from insightforge.agents.report.simple import SimpleReportGenerator

__all__ = ["LlmReportGenerator", "ReportGenerator", "SimpleReportGenerator"]
