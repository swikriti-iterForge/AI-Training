"""Settings, read from environment variables or the project's .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PREFERENCE_FILE = (
    Path(__file__).resolve().parents[4] / "data" / "memory" / "preferences.json"
)
DEFAULT_CONVERSATION_DIR = Path(__file__).resolve().parents[4] / "data" / "memory" / "chats"

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
    # How the search ranks: vector | keyword | hybrid. Change to hybrid only after
    # `python eval/check_queries.py --compare` shows it is clearly better on the real model.
    search_mode: str = "vector"
    # Below this best-match similarity the agent does not recommend anything and asks for more detail.
    # PROVISIONAL: calibrate with `python eval/check_cutoff.py`, then set TRAILMATE_MIN_RELEVANCE.
    min_relevance: float = 0.25
    # The recommendation text is built by code from the retrieved facts: instant, and it cannot invent
    # anything. True = let the model write it instead (opt-in; slower, and it made false claims in testing).
    explain_with_llm: bool = False
    # Long-term memory: what users asked us to remember (a plain JSON file; git-ignored)
    preference_file: Path = DEFAULT_PREFERENCE_FILE
    # Learn from what users ask for (per user id). Turn off with TRAILMATE_LEARN_AUTOMATICALLY=false.
    learn_automatically: bool = True
    # How often something must be asked for before it counts as a USUAL preference. 2 is a starting guess.
    min_mentions: int = 2
    # What 'short' and 'long' mean, in km. From this data: every Easy trail is <= 4.0 and every Hard trail >= 8.2.
    short_max_km: float = 4.0
    long_min_km: float = 8.0
    # Saved chat history: one file per conversation (git-ignored)
    conversation_dir: Path = DEFAULT_CONVERSATION_DIR
    # Optional JSON file overriding guardrail rules (see guardrails.GuardrailConfig)
    guardrail_config_path: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
