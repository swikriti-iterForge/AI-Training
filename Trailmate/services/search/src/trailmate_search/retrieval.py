"""The three ways to rank trails, and how to combine them.

  vector   compares MEANING (cosine distance between embeddings). Finds "waterfall views without the
           crowds" for "quiet waterfall hike", but is weak on exact constraints.
  keyword  compares WORDS (PostgreSQL full-text search with stemming). Exact, but blind to synonyms.
           Note: PostgreSQL's ts_rank has no IDF, so it is cruder than BM25.
  hybrid   Reciprocal Rank Fusion: each trail scores 1/(k + rank) in each list; the scores add up.
           A trail that ranks well in BOTH lists wins. It uses ranks only, so the two different score
           scales never have to be compared.

All three keep ONE best chunk per trail, and share the same optional filters.
"""

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Boolean, ColumnClause, Select, func, literal_column, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from trailmate_search.models import Chunk, Trail

ENGLISH: ColumnClause[Any] = literal_column("'english'::regconfig")
RRF_K = 60  # the constant from the original RRF paper; larger = flatter weighting of top ranks
CANDIDATE_LIMIT = 50


@dataclass(frozen=True)
class Candidate:
    trail_id: int
    section: str
    text: str
    value: float  # cosine distance (vector) or ts_rank (keyword)


def keyword_tokens(query: str, limit: int = 12) -> list[str]:
    """Plain lowercase words. Only letters and digits survive, so the result is safe to hand to to_tsquery."""
    return list(dict.fromkeys(re.findall(r"[a-z0-9]+", query.lower())))[:limit]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion. Each ranking is an ordered list of ids, best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return scores


def _scope(stmt: Select, strategy: str, difficulty: list[str] | None, exclude: list[str]) -> Select:
    stmt = stmt.join(Trail, Trail.id == Chunk.trail_id).where(Chunk.strategy == strategy)
    if difficulty:
        stmt = stmt.where(Trail.difficulty.in_(difficulty))
    if exclude:
        stmt = stmt.where(Trail.slug.not_in(exclude))
    return stmt


async def vector_candidates(
    session: AsyncSession,
    query_vector: list[float],
    strategy: str,
    difficulty: list[str] | None,
    exclude: list[str],
) -> list[Candidate]:
    distance = Chunk.embedding.cosine_distance(query_vector)
    base = _scope(
        select(Chunk.trail_id, Chunk.section, Chunk.text, distance.label("value")),
        strategy,
        difficulty,
        exclude,
    )
    inner = base.ext(distinct_on(Chunk.trail_id)).order_by(Chunk.trail_id, distance).subquery()
    rows = (
        await session.execute(select(inner).order_by(inner.c.value).limit(CANDIDATE_LIMIT))
    ).all()
    return [Candidate(r.trail_id, r.section, r.text, float(r.value)) for r in rows]


async def keyword_candidates(
    session: AsyncSession,
    tokens: list[str],
    strategy: str,
    difficulty: list[str] | None,
    exclude: list[str],
) -> list[Candidate]:
    if not tokens:
        return []
    # OR between words: a trail matches if it has ANY of them; ts_rank prefers chunks with more of them.
    query = func.to_tsquery(ENGLISH, " | ".join(tokens))
    rank = func.ts_rank(Chunk.tsv, query)
    base = _scope(
        select(Chunk.trail_id, Chunk.section, Chunk.text, rank.label("value")),
        strategy,
        difficulty,
        exclude,
    )
    base = base.where(Chunk.tsv.op("@@")(query))
    inner = base.ext(distinct_on(Chunk.trail_id)).order_by(Chunk.trail_id, rank.desc()).subquery()
    rows = (
        await session.execute(select(inner).order_by(inner.c.value.desc()).limit(CANDIDATE_LIMIT))
    ).all()
    return [Candidate(r.trail_id, r.section, r.text, float(r.value)) for r in rows]


async def terms_found(
    session: AsyncSession, trail_ids: list[int], terms: list[str], strategy: str
) -> dict[int, list[str]]:
    """Which of the given words appear ANYWHERE in each trail's text. This is evidence, not ranking:
    the agent uses it to say honestly what a recommended trail does and does not mention."""
    words = [t for t in dict.fromkeys(terms) if re.fullmatch(r"[a-z0-9]+", t)][:8]
    if not words or not trail_ids:
        return {}
    columns = [
        func.bool_or(
            Chunk.tsv.op("@@", return_type=Boolean)(func.plainto_tsquery(ENGLISH, w))
        ).label(f"t{i}")
        for i, w in enumerate(words)
    ]
    stmt = (
        select(Chunk.trail_id, *columns)
        .where(Chunk.trail_id.in_(trail_ids), Chunk.strategy == strategy)
        .group_by(Chunk.trail_id)
    )
    return {
        row[0]: [w for i, w in enumerate(words) if row[i + 1]]
        for row in (await session.execute(stmt)).all()
    }
