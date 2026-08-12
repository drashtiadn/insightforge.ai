"""Tests for LangSmith tracing setup and debug helpers."""

from __future__ import annotations

import pytest

from insightforge.core.config import Settings, get_settings
from insightforge.graph import run_research
from insightforge.infrastructure.tracing import (
    configure_tracing,
    debug_hints,
    debug_run_summary,
    tracing_status,
)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_configure_tracing_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    status = configure_tracing(Settings(_env_file=None))

    assert status.enabled is False
    assert status.reason == "LANGSMITH_TRACING is false"
    assert tracing_status().enabled is False


def test_configure_tracing_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    status = configure_tracing(Settings(_env_file=None))

    assert status.enabled is False
    assert status.has_api_key is False
    assert "API_KEY" in status.reason


def test_configure_tracing_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "insightforge-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    status = configure_tracing(Settings(_env_file=None))

    assert status.enabled is True
    assert status.project == "insightforge-test"
    assert status.has_api_key is True
    assert status.reason == "configured"
    assert tracing_status().enabled is True


def test_debug_helpers() -> None:
    summary = debug_run_summary(
        phase="done",
        ok=True,
        score=1.0,
        errors=(),
        transitions=("init->plan", "evaluate->done"),
    )
    assert summary["phase"] == "done"
    assert summary["ok"] is True
    assert summary["error_count"] == 0
    assert summary["transitions"][-1] == "evaluate->done"

    hints = debug_hints()
    assert hints["ui"] == "https://smith.langchain.com"
    assert hints["run_name"] == "run_research"


def test_run_research_works_with_tracing_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Configure env so @traceable sees tracing flags, but do not require a live
    # LangSmith network call to succeed for this unit test.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    configure_tracing(Settings(_env_file=None))

    result = run_research("langsmith tracing", stub_search=True)
    assert result.ok
    assert result.phase == "done"
