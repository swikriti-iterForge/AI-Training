"""Tiny terminal client for the agent's streaming endpoint (until the real UI exists).

    python scripts/chat.py "I want an easy waterfall hike"
    python scripts/chat.py "hello" --model smart
    python scripts/chat.py "hello" --verbose      # also show path, timings, trace

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
    parser.add_argument("--url", default="http://localhost:8000/chat")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    payload = {"message": args.message, **({"model": args.model} if args.model else {})}
    request = urllib.request.Request(
        args.url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    text, event = "", ""
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
                elif event == "error":
                    print(f"\n[error] {data['message']}", file=sys.stderr)
                elif args.verbose and event in ("meta", "route", "done"):
                    print(f"\n[{event}] {json.dumps(data)}", end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
