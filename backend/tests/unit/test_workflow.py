"""Tests for workflow retry, recovery, and the run_research API."""

from __future__ import annotations

import pytest

from insightforge.graph import (
    call_with_retry,
    fetch_source,
    initial_state,
    retrieve_node,
    run_research,
)
from insightforge.graph import nodes as graph_nodes


def test_call_with_retry_succeeds_first_try() -> None:
    assert call_with_retry(lambda: "ok", max_attempts=3) == "ok"


def test_call_with_retry_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary")
        return "recovered"

    assert call_with_retry(flaky, max_attempts=3) == "recovered"
    assert attempts["count"] == 3


def test_call_with_retry_raises_after_exhausting_attempts() -> None:
    def always_fail() -> str:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError, match="down"):
        call_with_retry(always_fail, max_attempts=2)


def test_call_with_retry_does_not_retry_validation_errors() -> None:
    attempts = {"count": 0}

    def bad_input() -> str:
        attempts["count"] += 1
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        call_with_retry(bad_input, max_attempts=5)
    assert attempts["count"] == 1


def test_retrieve_retries_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def flaky_fetch(query: str, step: int) -> dict[str, str]:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise TimeoutError("blip")
        return fetch_source(query, step)

    monkeypatch.setattr(graph_nodes, "fetch_source", flaky_fetch)

    state = initial_state("AI", max_retries=3)
    state["step"] = 1
    state["phase"] = "research"
    result = retrieve_node(state)

    assert attempts["count"] == 2
    assert result["phase"] == "retrieve"
    assert "errors" not in result
    assert result["sources"][0]["title"].startswith("Source 1")


def test_retrieve_records_error_after_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(query: str, step: int) -> dict[str, str]:
        raise ConnectionError("offline")

    monkeypatch.setattr(graph_nodes, "fetch_source", always_fail)

    state = initial_state("AI", max_retries=2)
    state["step"] = 1
    state["phase"] = "research"
    result = retrieve_node(state)

    assert result["phase"] == "failed"
    assert result["errors"] == ["retrieve failed: offline"]


def test_run_research_happy_path() -> None:
    result = run_research("multi-agent systems")

    assert result.ok
    assert result.phase == "done"
    assert result.errors == ()
    assert "multi-agent systems" in result.report
    assert "init->plan" in result.transitions
    assert result.transitions[-1].endswith("->done")


def test_run_research_validation_failure() -> None:
    result = run_research("   ")

    assert not result.ok
    assert result.phase == "failed"
    assert result.errors == ("query must not be empty",)
    assert result.report == ""


def test_run_research_recovers_after_retrieve_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(query: str, step: int) -> dict[str, str]:
        raise TimeoutError("offline")

    monkeypatch.setattr(graph_nodes, "fetch_source", always_fail)

    result = run_research("partial recovery", max_steps=1, max_retries=2)

    assert result.phase == "done"
    assert not result.ok
    assert any("retrieve failed" in err for err in result.errors)
    assert "partial recovery" in result.report
    assert "Errors:" in result.report
    # Research notes were kept; sources could not be fetched.
    assert result.state["notes"]
    assert result.state["sources"] == []
