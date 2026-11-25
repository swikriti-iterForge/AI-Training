"""Load the trail files into PostgreSQL: parse -> chunk -> embed -> store.

    uv run python -m trailmate_search.ingest

Safe to run repeatedly: trails are matched by slug and their chunks are replaced, so you can change
the chunking and re-run without creating duplicates.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from trailmate_search.config import get_settings
from trailmate_search.db import SessionLocal, engine
from trailmate_search.embedder_client import Embedder, EmbedderUnavailable, HttpEmbedder
from trailmate_search.models import Chunk, Trail
from trailmate_search.parsing import ChunkDraft, ParsedTrail, chunk_by_section, load_trails

BATCH_SIZE = 32  # texts per call to the embedder


@dataclass(frozen=True)
class IngestResult:
    trails: int
    chunks: int


async def embed_in_batches(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors += await embedder.embed(texts[start : start + BATCH_SIZE])
    return vectors


async def ingest(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    data_dir: Path,
    embedding_dim: int,
    strategy: str = "section",
) -> IngestResult:
    trails: list[ParsedTrail] = load_trails(data_dir)
    drafts: list[tuple[ParsedTrail, ChunkDraft]] = [
        (trail, draft) for trail in trails for draft in chunk_by_section(trail)
    ]

    # Embed everything first: if the embedder fails, the database is left untouched.
    vectors = await embed_in_batches(embedder, [draft.embedded_text for _, draft in drafts])
    if any(len(v) != embedding_dim for v in vectors):
        raise ValueError(f"embedder returned vectors that are not {embedding_dim}-dimensional")

    async with session_factory() as session, session.begin():  # one transaction: all or nothing
        rows: dict[str, Trail] = {t.slug: t for t in await session.scalars(select(Trail))}
        for trail in trails:
            f = trail.facts
            values = dict(
                name=trail.name,
                tagline=trail.tagline,
                difficulty=f.difficulty,
                distance_km=f.distance_km,
                elevation_gain_m=f.elevation_gain_m,
                best_for=f.best_for,
                dogs_allowed=f.dogs_allowed,
                source_file=trail.source_file,
            )
            row = rows.get(trail.slug)
            if row is None:
                row = rows[trail.slug] = Trail(slug=trail.slug, **values)
                session.add(row)
            else:
                for key, value in values.items():
                    setattr(row, key, value)
        await session.flush()  # sends the INSERTs so every trail.id exists

        await session.execute(delete(Chunk).where(Chunk.strategy == strategy))
        for (trail, draft), vector in zip(drafts, vectors, strict=True):
            session.add(
                Chunk(
                    trail_id=rows[trail.slug].id,
                    strategy=strategy,
                    position=draft.position,
                    section=draft.section,
                    text=draft.text,
                    embedded_text=draft.embedded_text,
                    embedding=vector,
                )
            )
    return IngestResult(trails=len(trails), chunks=len(drafts))


async def main() -> int:
    settings = get_settings()
    embedder = HttpEmbedder(settings.embedder_url, settings.embedder_timeout_s)
    try:
        result = await ingest(
            SessionLocal,
            embedder,
            settings.data_dir,
            settings.embedding_dim,
            settings.default_strategy,
        )
    except EmbedderUnavailable as exc:
        print(
            f"Cannot reach the embedder at {settings.embedder_url}: {exc}\n"
            "Start it first (see README), then run this again.",
            file=sys.stderr,
        )
        return 1
    finally:
        await embedder.aclose()
        await engine.dispose()
    print(
        f"Ingested {result.trails} trails as {result.chunks} chunks ({settings.default_strategy} strategy)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
