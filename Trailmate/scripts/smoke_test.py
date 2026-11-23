"""Stage 0.5 - checks that everything TrailMate needs on THIS machine actually works.

Uses only the Python standard library, so it runs before anything is installed:
    python scripts/smoke_test.py            (or:  uv run scripts/smoke_test.py)

Each check prints PASS / FAIL / SKIP with a hint. Fix FAILs top to bottom, then re-run.
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LITELLM = os.environ.get("LITELLM_URL", "http://localhost:4000")
MODEL = os.environ.get("SMOKE_OLLAMA_MODEL", "qwen2.5:3b")
HF_URL = os.environ.get("HF_URL", "https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
CERT_HINT = (
    "This looks like the company proxy certificate problem (same as with uv). Ask IT for the "
    "root CA, or tell me and I'll give you the exact setting."
)


def http(url: str, payload: dict | None = None, timeout: float = 120.0, headers: dict | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.status, resp.read().decode()


def load_master_key() -> str | None:
    """LITELLM_MASTER_KEY from the environment, or from Trailmate/.env (created by start_litellm.py)."""
    if os.environ.get("LITELLM_MASTER_KEY"):
        return os.environ["LITELLM_MASTER_KEY"]
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    try:
        for line in open(env_file, encoding="utf-8"):
            if line.strip().startswith("LITELLM_MASTER_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return None


def check_memory() -> tuple[str, str]:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("pt", ctypes.c_ulonglong), ("pa", ctypes.c_ulonglong),
                            ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong), ("ve", ctypes.c_ulonglong)]

            s = MemStatus()
            s.l = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))  # type: ignore[attr-defined]
            total, avail = s.total / 2**30, s.avail / 2**30
        else:
            info = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in open("/proc/meminfo"))}
            total, avail = info["MemTotal"] / 2**20, info["MemAvailable"] / 2**20
    except Exception as exc:  # noqa: BLE001
        return SKIP, f"could not read memory ({exc})"
    detail = f"{avail:.1f} GB free of {total:.1f} GB"
    if avail < 6:
        return FAIL, f"{detail}. Too low for a 3B model plus Postgres. Close apps, and in Rancher Desktop turn Kubernetes OFF and limit the VM to ~4-6 GB."
    return PASS, detail


def check_ollama() -> tuple[str, str]:
    try:
        _, body = http(f"{OLLAMA}/api/tags", timeout=5)
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"Ollama not reachable at {OLLAMA} ({exc}). Install it from ollama.com and start it."
    names = [m["name"] for m in json.loads(body).get("models", [])]
    if MODEL not in names and f"{MODEL}:latest" not in names:
        return FAIL, f"Ollama is running but '{MODEL}' is not pulled. Run: ollama pull {MODEL}   (have: {names or 'none'})"
    try:
        _, out = http(f"{OLLAMA}/api/generate", {"model": MODEL, "prompt": "Say hello in five words.", "stream": False,
                                                  "options": {"num_predict": 24}})
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"model '{MODEL}' is listed but generating failed ({exc})"
    r = json.loads(out)
    tps = r["eval_count"] / (r["eval_duration"] / 1e9) if r.get("eval_duration") else 0
    return PASS, f"{MODEL} answered: {r['response'].strip()[:50]!r}  ({tps:.1f} tokens/sec on this machine)"


def check_litellm() -> tuple[str, str]:
    try:
        http(f"{LITELLM}/health/liveliness", timeout=5)
    except Exception as exc:  # noqa: BLE001
        return FAIL, (f"LiteLLM not reachable at {LITELLM} ({exc}). Start it:  python scripts/start_litellm.py")
    key = load_master_key()
    if not key:
        return FAIL, "LiteLLM is up but no LITELLM_MASTER_KEY found. Run `python scripts/start_litellm.py` once; it creates one in .env."
    try:
        _, out = http(f"{LITELLM}/v1/chat/completions", {"model": "fast", "max_tokens": 24,
                      "messages": [{"role": "user", "content": "Say hello in five words."}]},
                      headers={"Authorization": f"Bearer {key}"})
    except urllib.error.HTTPError as exc:
        hint = " The key in .env doesn't match the one LiteLLM started with: restart LiteLLM." if exc.code == 401 else ""
        return FAIL, f"LiteLLM is up but tier 'fast' failed: HTTP {exc.code} {exc.read().decode()[:200]}{hint}"
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"LiteLLM is up but tier 'fast' failed ({exc})"
    text = json.loads(out)["choices"][0]["message"]["content"].strip()
    return PASS, f"tier 'fast' answered: {text[:50]!r}"


def check_postgres() -> tuple[str, str]:
    if not shutil.which("docker"):
        return SKIP, "docker command not found (is Rancher Desktop running with the dockerd engine?)"
    cmd = ["docker", "compose", "exec", "-T", "db", "psql", "-U", "trailmate", "-d", "trailmate", "-tAc",
           "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname='vector';"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"could not run docker compose ({exc})"
    if out.returncode != 0:
        return FAIL, f"db container not ready. Run: docker compose up -d db   ({out.stderr.strip()[:150]})"
    version = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "?"
    return PASS, f"PostgreSQL up, pgvector extension version {version}"


def check_huggingface() -> tuple[str, str]:
    try:
        status, _ = http(HF_URL, timeout=15)
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        hint = CERT_HINT if "CERTIFICATE" in reason.upper() else "Check internet access."
        return FAIL, f"cannot reach huggingface.co ({reason}). The embedding model downloads from there. {hint}"
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"cannot reach huggingface.co ({exc})"
    return PASS, f"huggingface.co reachable (HTTP {status}); the embedding model can be downloaded"


def check_node() -> tuple[str, str]:
    node = shutil.which("node")
    if not node:
        return FAIL, "Node.js not found. Needed for the frontend (Stage 4). Install the LTS version from nodejs.org."
    version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    return PASS, f"Node {version}"


CHECKS = [
    ("Free memory", check_memory),
    ("Ollama + model", check_ollama),
    ("LiteLLM proxy", check_litellm),
    ("Postgres + pgvector", check_postgres),
    ("Hugging Face access", check_huggingface),
    ("Node.js", check_node),
]


def main() -> int:
    failures = 0
    print("TrailMate environment smoke test\n")
    for name, fn in CHECKS:
        status, detail = fn()
        failures += status == FAIL
        print(f"[{status}] {name:<22} {detail}")
    print(f"\n{'All good.' if not failures else f'{failures} check(s) failed. Fix them top to bottom and re-run.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
