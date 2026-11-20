"""Day 9 - settings loaded from environment variables / an optional .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Defaults match `docker compose up -d db` (Postgres on host port 5433).
    database_url: str = "postgresql+asyncpg://bookverse:bookverse@localhost:5433/bookverse"
    test_database_url: str = "postgresql+asyncpg://bookverse:bookverse@localhost:5433/bookverse_test"
    # Dev-only default. Override JWT_SECRET_KEY in .env for anything real.
    jwt_secret_key: str = "dev-only-secret-change-me-in-dotenv-0123456789"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @property
    def sync_database_url(self) -> str:
        """Same DB with the sync driver - used by Alembic and the migration script."""
        return self.database_url.replace("+asyncpg", "+psycopg2")


@lru_cache  # read once, reuse
def get_settings() -> Settings:
    return Settings()
