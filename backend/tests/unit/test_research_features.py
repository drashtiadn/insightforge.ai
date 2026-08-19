"""Tests for Phase 6.2 research features (compression, aggregation, conflicts,
confidence, citations) and their integration into the Simple agents.
"""

from __future__ import annotations

from insightforge.agents import (
    SimplePlanner,
    SimpleReasoner,
    SimpleReflectionAgent,
    SimpleReportGenerator,
)
from insightforge.domain.models import (
    Conflict,
    Document,
    Evidence,
    EvidenceCluster,
    RetrievalHit,
)
from insightforge.research import (
    aggregate_evidence,
    build_citations,
    compress_documents,
    compress_hits,
    detect_conflicts,
    link_clusters_to_citations,
    reasoning_confidence,
    reflection_confidence,
)
from insightforge.shared.enums import RetrievalMode, SearchProviderHint


def _hit(hit_id: str, text: str, *, score: float = 0.8) -> RetrievalHit:
    return RetrievalHit(
        id=hit_id,
        score=score,
        text=text,
        metadata={
            "title": f"Doc {hit_id}",
            "url": f"https://example.com/{hit_id}",
        },
        mode=RetrievalMode.HYBRID,
    )


def _doc(title: str, url: str, snippet: str, *, score: float = 0.7) -> Document:
    return Document(
        title=title,
        url=url,
        snippet=snippet,
        provider=SearchProviderHint.WEB,
        score=score,
    )


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def test_compress_hits_dedupes_and_caps() -> None:
    hits = [
        _hit("h1", "Vector databases store embeddings for search."),
        _hit("h2", "Vector databases store embeddings for search."),
        _hit("h3", "  "),
        _hit("h4", "They power retrieval augmented generation."),
    ]
    kept = compress_hits(hits, max_items=2)
    assert [h.id for h in kept] == ["h1", "h4"]
    assert all(h.text for h in kept)


def test_compress_documents_dedupes_by_url() -> None:
    docs = [
        _doc("A", "https://example.com/a", "Alpha snippet."),
        _doc("A dup", "https://example.com/a", "Alpha snippet."),
        _doc("B", "https://example.com/b", "Beta snippet with more detail."),
    ]
    kept = compress_documents(docs)
    assert [d.title for d in kept] == ["A", "B"]


def test_compress_hits_respects_min_score() -> None:
    hits = [
        _hit("low", "irrelevant", score=0.1),
        _hit("high", "very relevant", score=0.9),
    ]
    kept = compress_hits(hits, min_score=0.5)
    assert [h.id for h in kept] == ["high"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_evidence_groups_similar_snippets() -> None:
    hits = [
        _hit("h1", "Vector databases store high-dimensional embeddings."),
        _hit("h2", "Vector databases keep embeddings for similarity search."),
        _hit("h3", "Airplanes fly through the sky above clouds."),
    ]
    clusters = aggregate_evidence(hits=hits)
    # First two hits belong together, airplane hit is its own cluster.
    biggest = max(clusters, key=lambda c: len(c.evidence))
    assert len(biggest.evidence) == 2
    assert any("airplanes" in c.claim.lower() for c in clusters)


def test_aggregate_evidence_handles_empty_inputs() -> None:
    assert aggregate_evidence() == []


def test_aggregate_evidence_dedupes_hit_and_document_same_url() -> None:
    """Pipeline used to pass retrieval hits and their source documents together."""

    text = "Hybrid RAG mixes lexical and dense retrieval signals."
    hits = [_hit("c0", text, score=0.95)]
    docs = [_doc("RAG", "https://example.com/c0", text, score=0.5)]
    clusters = aggregate_evidence(hits=hits, documents=docs)

    assert len(clusters) == 1
    assert len(clusters[0].evidence) == 1
    assert clusters[0].evidence[0].source_id == "https://example.com/c0"


def test_simple_reasoner_does_not_double_count_hit_and_document() -> None:
    text = "An overview of budget test covering definitions and practice."
    hit = _hit("c0", text, score=0.9)
    doc = _doc("Source 1", "https://example.com/c0", text)
    result = SimpleReasoner().reason("budget test", hits=[hit], documents=[doc])

    assert result.used_source_ids == ["https://example.com/c0"]
    reflection = SimpleReflectionAgent().reflect(reasoning=result, hits=[hit], documents=[doc])
    assert any("only 1 source" in gap for gap in reflection.gaps)
    assert reflection.follow_up_queries


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def test_detect_conflicts_flags_negation_mismatch() -> None:
    clusters = [
        EvidenceCluster(
            claim="Vaccines prevent measles infections effectively.",
            evidence=[Evidence(source_id="a", snippet="…", score=0.9)],
            score=0.9,
        ),
        EvidenceCluster(
            claim="Vaccines do not prevent measles infections effectively.",
            evidence=[Evidence(source_id="b", snippet="…", score=0.8)],
            score=0.8,
        ),
    ]
    conflicts = detect_conflicts(clusters)
    assert len(conflicts) == 1
    assert conflicts[0].sources_a == ["a"]
    assert conflicts[0].sources_b == ["b"]


def test_detect_conflicts_ignores_unrelated_claims() -> None:
    clusters = [
        EvidenceCluster(
            claim="Cats are common household pets.",
            evidence=[Evidence(source_id="a", snippet="…")],
        ),
        EvidenceCluster(
            claim="Quantum computers use qubits, not classical bits.",
            evidence=[Evidence(source_id="b", snippet="…")],
        ),
    ]
    assert detect_conflicts(clusters) == []


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def test_reasoning_confidence_bounds() -> None:
    clusters = [
        EvidenceCluster(
            claim="c1",
            evidence=[Evidence(source_id="a", snippet="…", score=0.8)],
            score=0.8,
        ),
        EvidenceCluster(
            claim="c2",
            evidence=[Evidence(source_id="b", snippet="…", score=0.7)],
            score=0.7,
        ),
    ]
    score = reasoning_confidence(clusters, [], source_count=2)
    assert 0.0 < score <= 1.0

    conflicts = [Conflict(claim_a="a", claim_b="b")]
    with_conflict = reasoning_confidence(clusters, conflicts, source_count=2)
    assert with_conflict < score

    assert reasoning_confidence([], [], source_count=0) == 0.0


def test_reflection_confidence_blends_signals() -> None:
    high = reflection_confidence(
        is_sufficient=True,
        source_count=4,
        conflict_count=0,
        reasoning=0.8,
    )
    low = reflection_confidence(
        is_sufficient=False,
        source_count=1,
        conflict_count=2,
        reasoning=0.3,
    )
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_build_and_link_citations() -> None:
    clusters = [
        EvidenceCluster(
            claim="Alpha claim",
            evidence=[
                Evidence(
                    source_id="s1",
                    snippet="alpha",
                    url="https://example.com/a",
                    title="A",
                ),
                Evidence(
                    source_id="s1-dup",
                    snippet="alpha again",
                    url="https://example.com/a",
                    title="A",
                ),
            ],
        ),
        EvidenceCluster(
            claim="Beta claim",
            evidence=[
                Evidence(
                    source_id="s2",
                    snippet="beta",
                    url="https://example.com/b",
                    title="B",
                ),
            ],
        ),
    ]
    citations = build_citations(clusters)
    assert len(citations) == 2
    assert citations[0].url == "https://example.com/a"
    assert citations[1].url == "https://example.com/b"

    links = link_clusters_to_citations(clusters, citations)
    assert links == [[1], [2]]


def test_build_citations_uses_locator_when_no_url() -> None:
    clusters = [
        EvidenceCluster(
            claim="local claim",
            evidence=[Evidence(source_id="chunk-42", snippet="local")],
        )
    ]
    citations = build_citations(clusters)
    assert citations[0].url is None
    assert citations[0].locator == "chunk chunk-42"


# ---------------------------------------------------------------------------
# Integration: SimpleReasoner / SimpleReflectionAgent / SimpleReportGenerator
# ---------------------------------------------------------------------------


def test_simple_reasoner_populates_clusters_and_conflicts() -> None:
    hits = [
        _hit("h1", "Coffee improves alertness in most adults.", score=0.9),
        _hit("h2", "Coffee improves alertness for tired workers.", score=0.85),
        _hit("h3", "Coffee does not improve alertness in most adults.", score=0.7),
    ]
    result = SimpleReasoner().reason("coffee alertness", hits=hits)

    assert result.clusters
    assert result.conflicts, "opposing claims should produce a conflict"
    assert result.answer
    assert result.used_source_ids
    assert 0.0 < result.confidence <= 1.0


def test_simple_reflection_flags_conflicts_and_emits_follow_ups() -> None:
    reasoner = SimpleReasoner()
    hits = [
        _hit("h1", "Coffee improves alertness in most adults.", score=0.9),
        _hit("h2", "Coffee does not improve alertness in most adults.", score=0.85),
    ]
    reasoning = reasoner.reason("coffee alertness", hits=hits)
    verdict = SimpleReflectionAgent().reflect(reasoning, hits=hits)

    assert verdict.conflicts
    assert not verdict.is_sufficient
    assert any("conflict" in gap.lower() for gap in verdict.gaps)
    assert verdict.follow_up_queries


def test_simple_report_includes_findings_evidence_and_citations() -> None:
    hits = [_hit("h1", "LangGraph orchestrates multi-agent workflows.", score=0.9)]
    doc = _doc(
        "LangGraph docs",
        "https://example.com/langgraph",
        "LangGraph is a framework for building multi-agent applications.",
    )

    plan = SimplePlanner().build_plan("multi-agent frameworks")
    reasoning = SimpleReasoner().reason(
        "multi-agent frameworks",
        hits=hits,
        documents=[doc],
    )
    reflection = SimpleReflectionAgent().reflect(reasoning, hits=hits, documents=[doc])
    report = SimpleReportGenerator().generate(
        plan=plan,
        reasoning=reasoning,
        reflection=reflection,
        hits=hits,
        documents=[doc],
    )

    headings = {section.heading for section in report.sections}
    assert {"Research Plan", "Summary", "Key Findings", "Evidence"}.issubset(headings)
    assert report.citations
    markdown = report.to_markdown()
    assert "## Key Findings" in markdown
    assert "## Evidence" in markdown
    assert "## Citations" in markdown


def test_simple_report_renders_conflicts_when_present() -> None:
    hits = [
        _hit("h1", "Coffee improves alertness in most adults.", score=0.9),
        _hit("h2", "Coffee does not improve alertness in most adults.", score=0.85),
    ]
    plan = SimplePlanner().build_plan("coffee alertness")
    reasoning = SimpleReasoner().reason("coffee alertness", hits=hits)
    report = SimpleReportGenerator().generate(plan=plan, reasoning=reasoning, hits=hits)

    headings = {section.heading for section in report.sections}
    assert "Conflicts" in headings
