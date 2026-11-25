"""Fixtures. Tests use a SEPARATE database (trailmate_search_test), created automatically, and a real
PostgreSQL with pgvector. If Postgres isn't reachable, the database tests are skipped with a message."""

import pytest
from fakes import FakeEmbedder
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from trailmate_search.config import get_settings
from trailmate_search.db import get_session
from trailmate_search.ingest import ingest
from trailmate_search.main import create_app
from trailmate_search.models import Base

SETTINGS = get_settings()
TEST_URL = SETTINGS.test_database_url
SYNC_URL = TEST_URL.replace("+asyncpg", "+psycopg2")


@pytest.fixture(scope="session")
def test_db():
    """Once per run: create the test database if needed, enable pgvector, build fresh tables."""
    url = make_url(SYNC_URL)
    try:
        admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            if not conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            ):
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
        admin.dispose()
    except OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable - run `docker compose up -d db` ({exc.orig})")
    engine = create_engine(SYNC_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()


@pytest.fixture
async def session_factory(test_db):
    engine = create_async_engine(TEST_URL, poolclass=NullPool)
    yield async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:  # empty every table so tests stay independent
        await conn.execute(text("TRUNCATE trails, chunks RESTART IDENTITY CASCADE"))
    await engine.dispose()


@pytest.fixture
async def ingested(session_factory):
    """The 16 real trail files, loaded into the test database with the fake embedder."""
    await ingest(session_factory, FakeEmbedder(), SETTINGS.data_dir, SETTINGS.embedding_dim)
    return session_factory


def make_client(session_factory, embedder):
    app = create_app(embedder=embedder)

    async def test_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = test_session
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client(ingested):
    async with make_client(ingested, FakeEmbedder()) as c:
        yield c
