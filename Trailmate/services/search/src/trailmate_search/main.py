"""trailmate-search: find trails by meaning, by words, or both.

    POST /search       {"query": "quiet waterfall hike", "top_k": 3, "mode": "hybrid", "difficulty": ["Easy"]}
    GET  /trails/{slug}                                               -> one trail's facts
    GET  /trails                                                      -> all trails
    GET  /health

How a search works:
  1. embed the query (embedder service)
  2. rank trails by meaning (vector), by words (keyword), or fuse both (hybrid, RRF)
  3. optional filters (difficulty, slugs to exclude) apply to every mode, before ranking
  4. return the top_k trails: facts, the matching text, and which of your `terms` each trail really mentions
`score` is ALWAYS the cosine similarity of the best-matching chunk, whatever the mode, so callers can
judge how close a match really is (for example, to refuse a weak one).
"""

import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from trailmate_search.config import Settings, get_settings
from trailmate_search.db import SessionDep
from trailmate_search.embedder_client import Embedder, EmbedderUnavailable, HttpEmbedder
from trailmate_search.models import Chunk, Trail
from trailmate_search.retrieval import (
    keyword_candidates,
    keyword_tokens,
    rrf,
    terms_found,
    vector_candidates,
)

Mode = Literal["vector", "keyword", "hybrid"]


class Facts(BaseModel):
    """Only fields that exist in the data. Anything else is 'not verified'."""

    difficulty: str
    distance_km: float
    elevation_gain_m: int
    best_for: list[str]
    dogs_allowed: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)
    strategy: str = Field(default="section", description="which chunking strategy to search")
    mode: Mode | None = Field(
        default=None, description="vector | keyword | hybrid (default: the service setting)"
    )
    difficulty: list[str] | None = Field(
        default=None, max_length=4, description="only trails with one of these ratings"
    )
    exclude: list[str] = Field(
        default_factory=list, max_length=50, description="trail slugs to leave out"
    )
    terms: list[str] = Field(
        default_factory=list, max_length=8, description="words to check for in each trail's text"
    )


class SearchResult(BaseModel):
    slug: str
    name: str
    score: float = Field(
        description="cosine similarity of the best-matching chunk (higher = closer)"
    )
    matched_section: str
    excerpt: str = Field(description="the original text of the best-matching chunk")
    facts: Facts
    vector_rank: int | None = Field(
        default=None, description="position in the meaning-based ranking"
    )
    keyword_rank: int | None = Field(
        default=None, description="position in the word-based ranking (None = no match)"
    )
    fused_score: float | None = Field(default=None, description="RRF score (hybrid mode only)")
    terms_found: list[str] = Field(
        default_factory=list, description="which requested terms the trail's text mentions"
    )


class SearchResponse(BaseModel):
    query: str
    strategy: str
    mode: Mode
    results: list[SearchResult]
    timing_ms: dict[str, float]


class TrailOut(BaseModel):
    slug: str
    name: str
    tagline: str | None
    facts: Facts


def facts_of(trail: Trail) -> Facts:
    return Facts(
        difficulty=trail.difficulty,
        distance_km=float(trail.distance_km),
        elevation_gain_m=trail.elevation_gain_m,
        best_for=trail.best_for,
        dogs_allowed=trail.dogs_allowed,
    )


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def create_app(embedder: Embedder | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if isinstance(app.state.embedder, HttpEmbedder):
            await app.state.embedder.aclose()

    app = FastAPI(title="TrailMate search", version="0.2.0", lifespan=lifespan)
    app.state.embedder = embedder or HttpEmbedder(
        settings.embedder_url, settings.embedder_timeout_s
    )

    @app.exception_handler(EmbedderUnavailable)
    async def embedder_down(_: Request, exc: EmbedderUnavailable) -> JSONResponse:
        return JSONResponse(
            {"error": {"status": 503, "message": "The embedding service is unavailable."}},
            status_code=503,
        )

    @app.get("/health")
    async def health(session: SessionDep) -> dict[str, int | str]:
        trails = await session.scalar(select(func.count()).select_from(Trail))
        chunks = await session.scalar(select(func.count()).select_from(Chunk))
        return {
            "status": "ok",
            "trails": trails or 0,
            "chunks": chunks or 0,
            "default_mode": settings.default_mode,
        }

    @app.post("/search", response_model=SearchResponse)
    async def search(
        body: SearchRequest, session: SessionDep, embedder: EmbedderDep
    ) -> SearchResponse:
        mode: Mode = body.mode or settings.default_mode  # type: ignore[assignment]
        if mode not in ("vector", "keyword", "hybrid"):
            raise HTTPException(500, f"Server default mode '{mode}' is not valid")
        started = time.perf_counter()
        [query_vector] = await embedder.embed([body.query])
        embedded = time.perf_counter()

        scope = (body.strategy, body.difficulty, body.exclude)
        by_meaning = await vector_candidates(session, query_vector, *scope)
        by_words = (
            await keyword_candidates(session, keyword_tokens(body.query), *scope)
            if mode != "vector"
            else []
        )
        vector = {c.trail_id: c for c in by_meaning}
        keyword = {c.trail_id: c for c in by_words}
        vector_rank = {c.trail_id: i for i, c in enumerate(by_meaning, 1)}
        keyword_rank = {c.trail_id: i for i, c in enumerate(by_words, 1)}

        fused: dict[int, float] = {}
        if mode == "vector":
            order = [c.trail_id for c in by_meaning]
        elif mode == "keyword":
            order = [c.trail_id for c in by_words]
        else:
            fused = rrf([[c.trail_id for c in by_meaning], [c.trail_id for c in by_words]])
            order = sorted(fused, key=lambda t: (-fused[t], vector_rank.get(t, 10**6)))
        top = order[: body.top_k]

        trails = (
            {t.id: t for t in await session.scalars(select(Trail).where(Trail.id.in_(top)))}
            if top
            else {}
        )
        found = await terms_found(session, top, body.terms, body.strategy)
        finished = time.perf_counter()

        results = []
        for trail_id in top:
            shown = (
                keyword[trail_id] if mode == "keyword" else vector[trail_id]
            )  # the chunk behind this result
            trail = trails[trail_id]
            results.append(
                SearchResult(
                    slug=trail.slug,
                    name=trail.name,
                    score=round(1 - vector[trail_id].value, 4),
                    matched_section=shown.section,
                    excerpt=shown.text,
                    facts=facts_of(trail),
                    vector_rank=vector_rank.get(trail_id),
                    keyword_rank=keyword_rank.get(trail_id),
                    fused_score=round(fused[trail_id], 5) if fused else None,
                    terms_found=found.get(trail_id, []),
                )
            )
        return SearchResponse(
            query=body.query,
            strategy=body.strategy,
            mode=mode,
            results=results,
            timing_ms={
                "embed": round((embedded - started) * 1000, 1),
                "search": round((finished - embedded) * 1000, 1),
                "total": round((finished - started) * 1000, 1),
            },
        )

    @app.get("/trails", response_model=list[TrailOut])
    async def list_trails(session: SessionDep) -> list[TrailOut]:
        rows = await session.scalars(select(Trail).order_by(Trail.name))
        return [
            TrailOut(slug=t.slug, name=t.name, tagline=t.tagline, facts=facts_of(t)) for t in rows
        ]

    @app.get("/trails/{slug}", response_model=TrailOut)
    async def get_trail(slug: str, session: SessionDep) -> TrailOut:
        trail = await session.scalar(select(Trail).where(Trail.slug == slug))
        if trail is None:
            raise HTTPException(404, f"No trail with slug '{slug}'")
        return TrailOut(
            slug=trail.slug, name=trail.name, tagline=trail.tagline, facts=facts_of(trail)
        )

    return app


app = create_app()  # uvicorn trailmate_search.main:app
