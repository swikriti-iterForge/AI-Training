"""Settings, read from TRAILMATE_SEARCH_* environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Trailmate/data/trails (this file is Trailmate/services/search/src/trailmate_search/config.py)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "trails"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRAILMATE_SEARCH_", env_file=".env", extra="ignore"
    )

    # `docker compose up -d db` publishes Postgres on host port 5434
    database_url: str = "postgresql+asyncpg://trailmate:trailmate@localhost:5434/trailmate"
    test_database_url: str = (
        "postgresql+asyncpg://trailmate:trailmate@localhost:5434/trailmate_search_test"
    )
    embedder_url: str = "http://localhost:8001"
    embedder_timeout_s: float = 30.0
    data_dir: Path = DEFAULT_DATA_DIR
    embedding_dim: int = 384  # must match the embedding model (all-MiniLM-L6-v2 = 384)
    default_strategy: str = "section"  # which chunking strategy to search (see chunks.strategy)
    default_mode: str = "vector"  # vector | keyword | hybrid. Switch to hybrid once eval/check_queries.py --compare says so

    @property
    def sync_database_url(self) -> str:
        """Same database with the sync driver. Alembic and test setup use it."""
        return self.database_url.replace("+asyncpg", "+psycopg2")


@lru_cache
def get_settings() -> Settings:
    return Settings()
