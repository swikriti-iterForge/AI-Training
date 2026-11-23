"""Start the LiteLLM proxy for TrailMate.

    python scripts/start_litellm.py

- First run creates a random LITELLM_MASTER_KEY in Trailmate/.env. (LiteLLM refuses to start
  without a master key, because otherwise anyone who can reach it could use your models.)
- The agent and scripts/smoke_test.py read the same .env, so there is one key in one place.
- It listens on 127.0.0.1 only, so other machines on your network cannot reach it.
- .env is git-ignored. Never commit it.
"""

import argparse
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
CONFIG = ROOT / "infra" / "litellm_config.yaml"


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def ensure_master_key(path: Path) -> tuple[str, bool]:
    """Return (key, created). Creates and saves a key if .env doesn't have one yet."""
    existing = read_env(path).get("LITELLM_MASTER_KEY")
    if existing:
        return existing, False
    key = "sk-" + secrets.token_hex(32)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + f"LITELLM_MASTER_KEY={key}\n", encoding="utf-8")
    return key, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the LiteLLM proxy")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--dry-run", action="store_true", help="show the command, don't start")
    args = parser.parse_args()

    uvx = shutil.which("uvx")
    if not uvx:
        print("uvx not found. It comes with uv: check that `uv --version` works.", file=sys.stderr)
        return 1

    key, created = ensure_master_key(ENV_FILE)
    if created:
        print(f"Created a new LITELLM_MASTER_KEY in {ENV_FILE}")
        print("If the agent is already running, restart it so it picks up the key.\n")

    command = [uvx, "--from", "litellm[proxy]", "litellm", "--config", str(CONFIG),
               "--host", "127.0.0.1", "--port", str(args.port)]
    print("Starting LiteLLM on http://127.0.0.1:%d  (Ctrl+C to stop)" % args.port)
    if args.dry_run:
        print(" ".join(command))
        return 0
    try:
        return subprocess.run(command, env={**os.environ, "LITELLM_MASTER_KEY": key}, cwd=ROOT).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
