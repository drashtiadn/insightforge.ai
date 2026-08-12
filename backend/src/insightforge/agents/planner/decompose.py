"""Decompose an analyzed query into ordered, provider-ready research tasks."""

from __future__ import annotations

from insightforge.agents.planner.schemas import QueryAnalysis, ResearchTask
from insightforge.core.logging import get_logger
from insightforge.shared.enums import QueryIntent, SearchProviderHint

logger = get_logger(__name__)


def _keywords_query(analysis: QueryAnalysis) -> str:
    """Prefer keyword phrase; fall back to the normalized query."""

    if analysis.keywords:
        return " ".join(analysis.keywords)
    return analysis.normalized_query


def _task(
    *,
    task_id: str,
    description: str,
    search_query: str,
    providers: list[SearchProviderHint],
    priority: int,
) -> ResearchTask:
    return ResearchTask(
        id=task_id,
        description=description,
        search_query=search_query,
        providers=providers,
        priority=priority,
    )


def decompose_tasks(analysis: QueryAnalysis, intent: QueryIntent) -> list[ResearchTask]:
    """Build an ordered task list tailored to the detected intent.

    Each task carries provider hints so Phase 3.2 can route searches without
    re-analyzing the query.
    """

    topic = analysis.normalized_query
    keywords = _keywords_query(analysis)

    tasks: list[ResearchTask]
    match intent:
        case QueryIntent.FACTUAL:
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Collect encyclopedic background on {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.WIKIPEDIA, SearchProviderHint.WEB],
                    priority=1,
                ),
                _task(
                    task_id="t2",
                    description=f"Find authoritative web sources about {topic}",
                    search_query=keywords,
                    providers=[SearchProviderHint.WEB],
                    priority=2,
                ),
                _task(
                    task_id="t3",
                    description=f"Scan community discussion on {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.REDDIT],
                    priority=3,
                ),
            ]
        case QueryIntent.COMPARATIVE:
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Research each side of the comparison: {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.WEB, SearchProviderHint.WIKIPEDIA],
                    priority=1,
                ),
                _task(
                    task_id="t2",
                    description=f"Find comparison articles and benchmarks for {topic}",
                    search_query=f"{keywords} comparison",
                    providers=[SearchProviderHint.WEB],
                    priority=2,
                ),
                _task(
                    task_id="t3",
                    description=f"Gather practitioner opinions on {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.REDDIT, SearchProviderHint.YOUTUBE],
                    priority=3,
                ),
            ]
        case QueryIntent.HOWTO:
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Find written guides and tutorials for {topic}",
                    search_query=f"{keywords} tutorial guide",
                    providers=[SearchProviderHint.WEB],
                    priority=1,
                ),
                _task(
                    task_id="t2",
                    description=f"Find video walkthroughs for {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.YOUTUBE],
                    priority=2,
                ),
                _task(
                    task_id="t3",
                    description=f"Locate example implementations for {topic}",
                    search_query=keywords,
                    providers=[SearchProviderHint.GITHUB, SearchProviderHint.WEB],
                    priority=3,
                ),
            ]
        case QueryIntent.NEWS:
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Find recent news coverage of {topic}",
                    search_query=f"{keywords} latest",
                    providers=[SearchProviderHint.WEB],
                    priority=1,
                ),
                _task(
                    task_id="t2",
                    description=f"Check discussion threads about {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.REDDIT],
                    priority=2,
                ),
                _task(
                    task_id="t3",
                    description=f"Find recent video coverage of {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.YOUTUBE],
                    priority=3,
                ),
            ]
        case QueryIntent.ACADEMIC:
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Search scholarly papers related to {topic}",
                    search_query=keywords,
                    providers=[SearchProviderHint.ARXIV, SearchProviderHint.WEB],
                    priority=1,
                ),
                _task(
                    task_id="t2",
                    description=f"Collect background context on {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.WIKIPEDIA, SearchProviderHint.WEB],
                    priority=2,
                ),
                _task(
                    task_id="t3",
                    description=f"Find secondary summaries of research on {topic}",
                    search_query=f"{keywords} review",
                    providers=[SearchProviderHint.WEB],
                    priority=3,
                ),
            ]
        case QueryIntent.CODE:
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Find relevant repositories for {topic}",
                    search_query=keywords,
                    providers=[SearchProviderHint.GITHUB],
                    priority=1,
                ),
                _task(
                    task_id="t2",
                    description=f"Find documentation and API references for {topic}",
                    search_query=f"{keywords} documentation",
                    providers=[SearchProviderHint.WEB],
                    priority=2,
                ),
                _task(
                    task_id="t3",
                    description=f"Find community Q&A and examples for {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.WEB, SearchProviderHint.REDDIT],
                    priority=3,
                ),
            ]
        case QueryIntent.EXPLORATORY:
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Build an overview of {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.WIKIPEDIA, SearchProviderHint.WEB],
                    priority=1,
                ),
                _task(
                    task_id="t2",
                    description=f"Survey diverse web sources on {topic}",
                    search_query=keywords,
                    providers=[SearchProviderHint.WEB],
                    priority=2,
                ),
                _task(
                    task_id="t3",
                    description=f"Sample community and media takes on {topic}",
                    search_query=topic,
                    providers=[
                        SearchProviderHint.REDDIT,
                        SearchProviderHint.YOUTUBE,
                    ],
                    priority=3,
                ),
            ]
        case _:
            # Defensive default — should be unreachable with QueryIntent enum.
            tasks = [
                _task(
                    task_id="t1",
                    description=f"Research {topic}",
                    search_query=topic,
                    providers=[SearchProviderHint.WEB],
                    priority=1,
                ),
            ]

    logger.debug(
        "decomposed intent=%s task_count=%d",
        intent.value,
        len(tasks),
    )
    return tasks
