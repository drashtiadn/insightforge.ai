"""LLM client contract shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypedDict

from insightforge.shared.enums import LlmProviderHint


class LlmMessage(TypedDict):
    """One chat message (``system``, ``user``, or ``assistant``)."""

    role: str
    content: str


class LlmClient(ABC):
    """One text-generation backend.

    Transport and API failures should raise ``ExternalServiceError`` so
    agents can fall back to heuristic implementations.
    """

    name: LlmProviderHint

    @property
    def available(self) -> bool:
        """True when required config is present and the provider can run."""

        return True

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier used for generation calls."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant text for ``messages``."""
