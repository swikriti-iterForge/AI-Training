"""Shared fixtures. API tests use a SEPARATE database (bookverse_test), created automatically,
and call the app in memory (no server needed). If Postgres isn't running, those tests skip."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from bookverse.core.config import get_settings
from bookverse.db import get_session
from bookverse.main import app
from bookverse.models import Base

TEST_URL = get_settings().test_database_url
SYNC_URL = TEST_URL.replace("+asyncpg", "+psycopg2")


@pytest.fixture(scope="session")
def test_db():
    """Once per run: create the test database if needed, with fresh tables."""
    url = make_url(SYNC_URL)
    try:
        admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            if not conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}):
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
        admin.dispose()
    except OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable - start it with `docker compose up -d db` ({exc.orig})")
    engine = create_engine(SYNC_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()


@pytest.fixture
async def client(test_db):
    engine = create_async_engine(TEST_URL, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def test_session():  # swap the real DB for the test DB
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = test_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    async with engine.begin() as conn:  # empty every table so tests stay independent
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    await engine.dispose()
