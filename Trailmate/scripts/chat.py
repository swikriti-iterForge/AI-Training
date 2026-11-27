"""Tiny terminal client for the agent's streaming endpoint (until the real UI exists).

    python scripts/chat.py "hello" --verbose                      # one message, Phase 1 workflow
    python scripts/chat.py "easy waterfall hike" --mode agent     # one message, Phase 3 agent
    python scripts/chat.py -i                                     # a CONVERSATION (agent mode, with memory)
    python scripts/chat.py -i --user alice                        # ...as a named user, so "remember..." is saved

In -i mode every line you type is a new turn in the same session, so "show me more", "the second one" and
"something harder" work. Type  quit  to leave. --verbose also shows path, tool calls, timings and the trace.
Reads the Server-Sent Events stream and prints tokens as they arrive.
"""

import argparse
import json
import sys
import urllib.request


def run_turn(url: str, payload: dict, verbose: bool) -> tuple[str | None, list[dict]]:
    """Send one message, print the reply as it streams. Returns (session_id, trails)."""
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    event, session_id, trails = "", None, []
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        for raw in response:  # one line at a time, as the server sends it
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "meta":
                    session_id = data.get("session_id")
                if event == "token":
                    print(data["text"], end="", flush=True)
                elif event == "replace":  # a guardrail swapped the text already shown
                    print(f"\n[guardrail replaced the reply]\n{data['text']}", end="", flush=True)
                elif event == "tool":  # agent mode: what the agent is doing
                    print(f"\n[{data['label']}]", file=sys.stderr, flush=True)
                elif event == "results":  # agent mode: the trail cards, straight from the database
                    trails = data["trails"]
                elif event == "error":
                    print(f"\n[error] {data['message']}", file=sys.stderr)
                elif verbose and event in ("meta", "route", "done"):
                    print(f"\n[{event}] {json.dumps(data)}", end="", flush=True)
    print()
    if trails:
        print("\nTrails found (facts from the database):")
        for i, t in enumerate(trails, 1):
            f = t["facts"]
            print(f"  {i}. {t['name']}  score {t['score']:.3f} | {f['difficulty']} | {f['distance_km']:g} km | {f['elevation_gain_m']} m climb")
    return session_id, trails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?", help="the message (omit with -i)")
    parser.add_argument("-i", "--interactive", action="store_true", help="a conversation: one session, many turns")
    parser.add_argument("--model", help="model tier, e.g. fast or smart")
    parser.add_argument("--mode", choices=["workflow", "agent"], help="workflow = Phase 1 (default), agent = Phase 3 (default with -i)")
    parser.add_argument("--user", help="user id; needed for long-term preferences (default with -i: 'me')")
    parser.add_argument("--session", help="continue an existing session id")
    parser.add_argument("--url", default="http://localhost:8000/chat")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.interactive and not args.message:
        parser.error("give a message, or use -i for a conversation")

    mode = args.mode or ("agent" if args.interactive else "workflow")
    user = args.user or ("me" if args.interactive else None)
    session = args.session

    def payload(message: str) -> dict:
        body: dict = {"message": message, "mode": mode}
        for key, value in (("model", args.model), ("user_id", user), ("session_id", session)):
            if value:
                body[key] = value
        return body

    if not args.interactive:
        run_turn(args.url, payload(args.message), args.verbose)
        return 0

    print(f"TrailMate conversation (mode {mode}, user {user}). Type 'quit' to leave.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in {"quit", "exit"}:
            return 0
        if not line:
            continue
        print()
        session, _ = run_turn(args.url, payload(line), args.verbose)
        session = session or args.session
        print()


if __name__ == "__main__":
    sys.exit(main())
