"""Settings, read from TRAILMATE_* environment variables or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAILMATE_", env_file=".env", extra="ignore")

    # The LiteLLM proxy speaks the OpenAI API, so we only need a base URL.
    litellm_base_url: str = "http://localhost:4000/v1"
    litellm_api_key: str = "not-needed-locally"  # the local proxy runs without auth
    # Named model tiers (defined in infra/litellm_config.yaml). The app never names a real model.
    tiers: list[str] = ["fast", "smart"]
    default_tier: str = "fast"
    llm_timeout_s: float = 90.0  # CPU-only models can be slow
    # Optional JSON file overriding guardrail rules (see guardrails.GuardrailConfig)
    guardrail_config_path: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
