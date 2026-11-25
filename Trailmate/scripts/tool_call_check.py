"""Phase 3 pre-check: can the local models decide WHEN to call a search tool?

An agent works by letting the model choose: answer, ask a question, or call a tool. Small CPU models
are often weak at that, so we measure it before building the agent around it.

    python scripts/tool_call_check.py                    # tier "fast"
    python scripts/tool_call_check.py --models fast,smart

Needs Ollama and the LiteLLM proxy running (python scripts/start_litellm.py). Standard library only.
It does NOT run the search; it only checks whether the model asks for it correctly.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = (
    "You are TrailMate, a hiking assistant with a trail search tool. "
    "Call search_trails when the user wants trail recommendations or asks about specific trails. "
    "Do not call it for greetings, thanks, or questions about what you can do. "
    "If a request is too vague to search, ask ONE short clarifying question instead."
)

TOOL = {
    "type": "function",
    "function": {
        "name": "search_trails",
        "description": "Search the trail database by meaning. Returns matching trails with their facts.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What the user wants, in plain words"}},
            "required": ["query"],
        },
    },
}

# (prompt, expected). "either" prompts are shown but not scored: the right behaviour is debatable.
PROMPTS = [
    ("I want an easy waterfall hike", "tool"),
    ("Which trails are good for dogs?", "tool"),
    ("Looking for a hard scramble with a view", "tool"),
    ("quiet version of the popular waterfall hike", "tool"),
    ("Tell me about Cascade Falls Loop", "tool"),
    ("Hello!", "no_tool"),
    ("What can you do?", "no_tool"),
    ("Thanks, that was helpful", "no_tool"),
    ("Suggest a good trek", "either"),  # vague: ideally ask what kind
    ("Any hikes near Mumbai?", "either"),  # the data has no locations
]


def load_key() -> str:
    if os.environ.get("LITELLM_MASTER_KEY"):
        return os.environ["LITELLM_MASTER_KEY"]
    try:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LITELLM_MASTER_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


def ask(url: str, key: str, tier: str, prompt: str) -> tuple[dict, float]:
    payload = {
        "model": tier, "temperature": 0, "max_tokens": 150, "tool_choice": "auto", "tools": [TOOL],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        body = json.loads(response.read())
    return body["choices"][0]["message"], time.perf_counter() - start


def describe(message: dict) -> tuple[str, str, bool]:
    """-> (decision, shown text, arguments_valid). decision: tool | no_tool | broken."""
    calls = message.get("tool_calls") or []
    content = (message.get("content") or "").strip()
    if calls:
        fn = calls[0]["function"]
        try:
            args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
            valid = fn["name"] == "search_trails" and isinstance(args.get("query"), str) and bool(args["query"].strip())
        except (ValueError, TypeError, AttributeError):
            return "tool", f"{fn.get('name')}(<unreadable arguments>)", False
        return "tool", f"{fn['name']}(query={args.get('query')!r})", valid
    if "search_trails" in content or content.startswith("{"):  # tool call written out as plain text
        return "broken", f"tool call written as text: {content[:60]!r}", False
    return "no_tool", f"said: {content[:70]!r}", True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="fast", help="comma-separated tiers, e.g. fast,smart")
    parser.add_argument("--url", default="http://localhost:4000/v1/chat/completions")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = load_key()
    if not key:
        print("No LITELLM_MASTER_KEY found. Run `python scripts/start_litellm.py` once; it creates one in .env.")
        return 2

    for tier in [t.strip() for t in args.models.split(",") if t.strip()]:
        print(f"\n=== tier '{tier}' ===")
        correct = scored = calls = valid_calls = broken = 0
        times: list[float] = []
        for prompt, expected in PROMPTS:
            try:
                message, seconds = ask(args.url, key, tier, prompt)
            except urllib.error.HTTPError as exc:
                print(f"HTTP {exc.code} from LiteLLM: {exc.read().decode()[:200]}")
                if exc.code == 401:
                    print("The key in .env does not match the running proxy. Restart LiteLLM with the script.")
                return 2
            except (urllib.error.URLError, OSError) as exc:
                print(f"Cannot reach LiteLLM at {args.url}: {exc}\nStart it: python scripts/start_litellm.py")
                return 2
            decision, shown, valid = describe(message)
            times.append(seconds)
            calls += decision == "tool"
            valid_calls += decision == "tool" and valid
            broken += decision == "broken"
            if expected == "either":
                verdict = "info"
            else:
                scored += 1
                ok = decision == expected
                correct += ok
                verdict = "OK  " if ok else "MISS"
            print(f"[{verdict}] {prompt!r:<50} expected {expected:<8} -> {shown}   ({seconds:.1f}s)")
        print(f"\nRight decision (call the tool or not): {correct}/{scored}")
        print(f"Tool calls with valid arguments: {valid_calls}/{calls}   | tool call written as text instead: {broken}")
        print(f"Mean time per request: {sum(times) / len(times):.1f}s")
    print("\nSmall sample, one run, temperature 0. Treat it as a first signal, not a benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
