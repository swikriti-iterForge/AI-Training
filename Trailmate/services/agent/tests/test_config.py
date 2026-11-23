"""The agent and the LiteLLM proxy share one key, read from the environment or .env."""

from trailmate_agent.config import Settings


def clear(monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.delenv("TRAILMATE_LITELLM_API_KEY", raising=False)


def test_key_defaults_to_a_placeholder(monkeypatch):
    clear(monkeypatch)
    assert Settings(_env_file=None).litellm_api_key == "not-set"


def test_agent_reads_the_proxys_master_key(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-shared")
    assert Settings(_env_file=None).litellm_api_key == "sk-shared"


def test_agent_specific_variable_wins(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-shared")
    monkeypatch.setenv("TRAILMATE_LITELLM_API_KEY", "sk-override")
    assert Settings(_env_file=None).litellm_api_key == "sk-override"


def test_key_is_read_from_an_env_file(monkeypatch, tmp_path):
    clear(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("LITELLM_MASTER_KEY=sk-from-file\n", encoding="utf-8")
    assert Settings(_env_file=env).litellm_api_key == "sk-from-file"
