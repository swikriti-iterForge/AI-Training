"""Settings, read from environment variables or the project's .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Trailmate/.env: shared with scripts/start_litellm.py, so the proxy and the agent use ONE key.
ROOT_ENV = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRAILMATE_",
        env_file=(ROOT_ENV, ".env"),
        extra="ignore",
        populate_by_name=True,
    )

    # The LiteLLM proxy speaks the OpenAI API, so we only need a base URL and a key.
    litellm_base_url: str = "http://localhost:4000/v1"
    # Must equal the proxy's LITELLM_MASTER_KEY. `python scripts/start_litellm.py` creates it
    # in .env; TRAILMATE_LITELLM_API_KEY overrides it if you ever need a different one.
    litellm_api_key: str = Field(
        default="not-set",
        validation_alias=AliasChoices("TRAILMATE_LITELLM_API_KEY", "LITELLM_MASTER_KEY"),
    )
    # Named model tiers (defined in infra/litellm_config.yaml). The app never names a real model.
    tiers: list[str] = ["fast", "smart"]
    default_tier: str = "fast"
    llm_timeout_s: float = 90.0  # CPU-only models can be slow
    # trailmate-search: the tool the agent calls
    search_url: str = "http://localhost:8002"
    search_timeout_s: float = 20.0
    # Two results closer than this in score count as a tie, and the agent compares them instead of picking.
    # 0.01 is a starting guess from 12 test queries, not a tuned value.
    tie_margin: float = 0.01
    # False = write the recommendation from a template instead of asking the model (faster, less natural)
    explain_with_llm: bool = True
    # Optional JSON file overriding guardrail rules (see guardrails.GuardrailConfig)
    guardrail_config_path: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
