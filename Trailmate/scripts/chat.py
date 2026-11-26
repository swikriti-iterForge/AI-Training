"""Tiny terminal client for the agent's streaming endpoint (until the real UI exists).

    python scripts/chat.py "I want an easy waterfall hike"
    python scripts/chat.py "hello" --model smart
    python scripts/chat.py "hello" --verbose      # also show path, timings, trace
    python scripts/chat.py "easy waterfall hike" --mode agent    # Phase 3: the model decides, real trail data

Reads the Server-Sent Events stream and prints tokens as they arrive.
"""

import argparse
import json
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--model", help="model tier, e.g. fast or smart")
    parser.add_argument("--mode", choices=["workflow", "agent"], default="workflow",
                        help="workflow = Phase 1 (default), agent = Phase 3")
    parser.add_argument("--url", default="http://localhost:8000/chat")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    payload = {"message": args.message, "mode": args.mode, **({"model": args.model} if args.model else {})}
    request = urllib.request.Request(
        args.url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    text, event, trails = "", "", []
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        for raw in response:  # one line at a time, as the server sends it
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "token":
                    text += data["text"]
                    print(data["text"], end="", flush=True)
                elif event == "replace":  # a guardrail swapped the text already shown
                    text = data["text"]
                    print(f"\n[guardrail replaced the reply]\n{text}", end="", flush=True)
                elif event == "tool":  # agent mode: what the agent is doing
                    print(f"\n[{data['label']}]", file=sys.stderr, flush=True)
                elif event == "results":  # agent mode: the trail cards, straight from the database
                    trails = data["trails"]
                elif event == "error":
                    print(f"\n[error] {data['message']}", file=sys.stderr)
                elif args.verbose and event in ("meta", "route", "done"):
                    print(f"\n[{event}] {json.dumps(data)}", end="", flush=True)
    print()
    if trails:
        print("\nTrails found (facts from the database):")
        for i, t in enumerate(trails, 1):
            f = t["facts"]
            print(f"  {i}. {t['name']}  score {t['score']:.3f} | {f['difficulty']} | {f['distance_km']:g} km | {f['elevation_gain_m']} m climb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
