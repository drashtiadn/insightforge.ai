"""Tests for workflow retry, recovery, and the run_research API."""

from __future__ import annotations

import pytest

from insightforge.domain.models import Document
from insightforge.graph import (
    call_with_retry,
    fetch_source,
    initial_state,
    resolve_search_service,
    run_research,
    search_node,
)
from insightforge.graph import nodes as graph_nodes
from insightforge.graph.retry import _backoff_delay
from insightforge.infrastructure.search import SearchProvider, SearchService
from insightforge.shared.enums import SearchProviderHint


def test_call_with_retry_succeeds_first_try() -> None:
    assert call_with_retry(lambda: "ok", max_attempts=3) == "ok"


def test_call_with_retry_retries_then_succeeds() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary")
        return "recovered"

    assert (
        call_with_retry(flaky, max_attempts=3, base_delay=0.1, sleep=sleeps.append) == "recovered"
    )
    assert attempts["count"] == 3
    assert len(sleeps) == 2
    assert all(delay >= 0 for delay in sleeps)


def test_call_with_retry_raises_after_exhausting_attempts() -> None:
    sleeps: list[float] = []

    def always_fail() -> str:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError, match="down"):
        call_with_retry(always_fail, max_attempts=2, sleep=sleeps.append)
    assert len(sleeps) == 1


def test_call_with_retry_does_not_retry_validation_errors() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def bad_input() -> str:
        attempts["count"] += 1
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        call_with_retry(bad_input, max_attempts=5, sleep=sleeps.append)
    assert attempts["count"] == 1
    assert sleeps == []


def test_backoff_delay_grows_exponentially_with_jitter_bounds() -> None:
    # Without randomness we only assert the upper bound of full jitter.
    for attempt in (1, 2, 3, 4):
        delay = _backoff_delay(attempt, base_delay=0.1, max_delay=1.0)
        assert 0.0 <= delay <= min(1.0, 0.1 * (2 ** (attempt - 1)))


def test_search_retries_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = search_node(state)

    assert attempts["count"] == 2
    assert result["phase"] == "search"
    assert "errors" not in result
    assert result["sources"][0]["title"].startswith("Source 1")


def test_search_records_error_after_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(query: str, step: int) -> dict[str, str]:
        raise ConnectionError("offline")

    monkeypatch.setattr(graph_nodes, "fetch_source", always_fail)

    state = initial_state("AI", max_retries=2)
    state["step"] = 1
    state["phase"] = "research"
    result = search_node(state)

    assert result["phase"] == "failed"
    assert result["errors"] == ["search failed: offline"]


def test_run_research_happy_path() -> None:
    result = run_research("multi-agent systems", stub_search=True)

    assert result.ok
    assert result.phase == "done"
    assert result.errors == ()
    assert "multi-agent systems" in result.report.lower()
    assert result.state["step"] >= 3
    assert len(result.state["tasks"]) >= 3
    assert "init->plan" in result.transitions
    assert "research->search" in result.transitions
    assert result.transitions[-1].endswith("->done")
    assert "##" in result.report
    assert "## Evaluation" in result.report
    assert result.state["evaluation"]
    assert 0.0 <= result.confidence <= 1.0


def test_run_research_validation_failure() -> None:
    result = run_research("   ", stub_search=True)

    assert not result.ok
    assert result.phase == "failed"
    assert result.errors == ("query must not be empty",)
    assert result.report == ""


def test_run_research_recovers_after_search_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(query: str, step: int) -> dict[str, str]:
        raise TimeoutError("offline")

    monkeypatch.setattr(graph_nodes, "fetch_source", always_fail)

    result = run_research(
        "partial recovery",
        max_steps=1,
        max_retries=2,
        stub_search=True,
    )

    assert result.phase == "done"
    assert not result.ok
    assert any("search failed" in err for err in result.errors)
    assert "partial recovery" in result.report.lower()
    assert "## Errors" in result.report
    # Research notes were kept; sources could not be fetched.
    assert result.state["notes"]
    assert result.state["sources"] == []


def test_run_research_injects_and_closes_default_search_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = {"count": 0}

    class _Provider(SearchProvider):
        name = SearchProviderHint.WEB

        def search(self, query: str, *, limit: int = 5) -> list[Document]:
            return [
                Document(
                    title="Injected",
                    url="https://example.com/injected",
                    snippet=f"Detailed notes about {query} covering methods and results.",
                    content=f"Detailed notes about {query} covering methods, results, and limitations.",
                    provider=SearchProviderHint.WEB,
                )
            ]

    class _OwnedService(SearchService):
        def close(self) -> None:
            closed["count"] += 1
            super().close()

    service = _OwnedService(
        {
            SearchProviderHint.WEB: _Provider(),
            SearchProviderHint.WIKIPEDIA: _Provider(),
            SearchProviderHint.ARXIV: _Provider(),
            SearchProviderHint.GITHUB: _Provider(),
            SearchProviderHint.YOUTUBE: _Provider(),
            SearchProviderHint.REDDIT: _Provider(),
        },
        scoring=True,
        client=None,
        owns_client=False,
    )

    monkeypatch.setattr(
        "insightforge.graph.workflow.create_search_service",
        lambda: service,
    )

    result = run_research("climate models", max_steps=1)

    assert result.ok
    assert result.phase == "done"
    assert any(doc.get("title") == "Injected" for doc in result.state["documents"])
    assert closed["count"] == 1


def test_resolve_search_service_stub_and_owned() -> None:
    stub, owns_stub = resolve_search_service(stub_search=True)
    assert stub is None
    assert owns_stub is False

    explicit = SearchService({})
    resolved, owns = resolve_search_service(explicit)
    assert resolved is explicit
    assert owns is False
