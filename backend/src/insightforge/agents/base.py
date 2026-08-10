"""Base contract shared by every research agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from insightforge.agents.tools import Tool


class BaseAgent(ABC):
    """Common shape for agents in the research pipeline.

    Subclasses set ``name`` and implement ``run``. Optional ``tools`` let an
    agent call shared capabilities (search, fetch, etc.) without hard-wiring
    infrastructure into the agent itself.
    """

    name: str

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.tools = list(tools or [])

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's main work."""

    def get_tool(self, name: str) -> Tool:
        """Return a registered tool by name.

        Raises:
            KeyError: If no tool with that name is registered.
        """

        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(f"tool not found: {name}")
