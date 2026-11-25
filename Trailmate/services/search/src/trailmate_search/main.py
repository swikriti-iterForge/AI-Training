"""trailmate-search: find trails by meaning.

    POST /search       {"query": "quiet waterfall hike", "top_k": 3}  -> trails, best match first
    GET  /trails/{slug}                                               -> one trail's facts
    GET  /trails                                                      -> all trails
    GET  /health

How a search works:
  1. embed the query (embedder service)
  2. compare it with every chunk's vector using cosine distance (pgvector's `<=>`)
  3. keep only the best chunk per trail, so results are distinct trails, not fragments of one trail
  4. return the top_k trails with their facts and the text that matched
"""

import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import distinct_on

from trailmate_search.config import Settings, get_settings
from trailmate_search.db import SessionDep
from trailmate_search.embedder_client import Embedder, EmbedderUnavailable, HttpEmbedder
from trailmate_search.models import Chunk, Trail


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


class SearchResult(BaseModel):
    slug: str
    name: str
    score: float = Field(
        description="cosine similarity of the best-matching chunk (higher = closer)"
    )
    matched_section: str
    excerpt: str = Field(description="the original text of the best-matching chunk")
    facts: Facts


class SearchResponse(BaseModel):
    query: str
    strategy: str
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

    app = FastAPI(title="TrailMate search", version="0.1.0", lifespan=lifespan)
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
        return {"status": "ok", "trails": trails or 0, "chunks": chunks or 0}

    @app.post("/search", response_model=SearchResponse)
    async def search(
        body: SearchRequest, session: SessionDep, embedder: EmbedderDep
    ) -> SearchResponse:
        started = time.perf_counter()
        [query_vector] = await embedder.embed([body.query])
        embedded = time.perf_counter()

        distance = Chunk.embedding.cosine_distance(query_vector)
        # DISTINCT ON (trail_id) + ORDER BY distance keeps the single closest chunk of each trail.
        best_per_trail = (
            select(Chunk.trail_id, Chunk.section, Chunk.text, distance.label("distance"))
            .where(Chunk.strategy == body.strategy)
            .ext(distinct_on(Chunk.trail_id))
            .order_by(Chunk.trail_id, distance)
            .subquery()
        )
        rows = (
            await session.execute(
                select(
                    Trail,
                    best_per_trail.c.section,
                    best_per_trail.c.text,
                    best_per_trail.c.distance,
                )
                .join(best_per_trail, best_per_trail.c.trail_id == Trail.id)
                .order_by(best_per_trail.c.distance)
                .limit(body.top_k)
            )
        ).all()
        finished = time.perf_counter()

        return SearchResponse(
            query=body.query,
            strategy=body.strategy,
            results=[
                SearchResult(
                    slug=trail.slug,
                    name=trail.name,
                    score=round(1 - float(dist), 4),
                    matched_section=section,
                    excerpt=text,
                    facts=facts_of(trail),
                )
                for trail, section, text, dist in rows
            ],
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
