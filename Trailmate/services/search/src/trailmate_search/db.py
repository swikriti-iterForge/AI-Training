"""Async database engine and a session per request (same pattern as BookVerse).

Vectors travel as text ("[0.1,0.2,...]") and Postgres converts them to its `vector` type.
That works with SQLAlchemy's pgvector `Vector` column as it is, with no extra driver setup.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from trailmate_search.config import get_settings


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


engine = make_engine(get_settings().database_url)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    async with SessionLocal() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
