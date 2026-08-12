"""Tool interface — small reusable capabilities agents can call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class Tool(ABC):
    """A single named capability (search, fetch URL, calculate, ...).

    Keep tools focused: one job per tool. Agents compose tools; they do not
    embed provider SDKs directly.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool with keyword arguments."""


class FunctionTool(Tool):
    """Wrap a plain Python function as a ``Tool``.

    Useful for stubs, tests, and thin adapters around infrastructure clients.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self._func = func

    def run(self, **kwargs: Any) -> Any:
        return self._func(**kwargs)
