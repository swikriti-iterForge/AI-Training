"""Search from the terminal. Needs the embedder (port 8001) and search service (port 8002) running.

    python scripts/search.py "quiet version of the popular waterfall hike"
    python scripts/search.py "easy dog friendly walk" --top-k 5
    python scripts/search.py "hard scramble with a waterfall" --json     # the raw JSON response
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Search TrailMate trails from the terminal")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3, help="how many trails (1 to 10)")
    parser.add_argument("--url", default="http://localhost:8002/search")
    parser.add_argument("--json", action="store_true", help="print the raw JSON instead")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe on any Windows console

    request = urllib.request.Request(
        args.url,
        json.dumps({"query": args.query, "top_k": args.top_k}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        hints = {422: "The request was rejected. Is --top-k between 1 and 10?",
                 503: "The search service cannot reach the embedder on port 8001. Is it running?"}
        print(f"Search service answered HTTP {exc.code}. {hints.get(exc.code, 'See its terminal for the error.')}")
        return 2
    except (urllib.error.URLError, OSError) as exc:
        print(f"Cannot reach the search service at {args.url}: {exc}\nStart it first (see the README).")
        return 2

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    t = data["timing_ms"]
    print(f"\nQuery: {data['query']}")
    print(f"({t['total']:.0f} ms total: embed {t['embed']:.0f}, database search {t['search']:.0f})\n")
    if not data["results"]:
        print("No results.")
    for i, r in enumerate(data["results"], 1):
        f = r["facts"]
        excerpt = " ".join(r["excerpt"].split())
        print(f"{i}. {r['name']}    score {r['score']:.3f}")
        print(f"   {f['difficulty']} | {f['distance_km']} km | {f['elevation_gain_m']} m climb | dogs: {f['dogs_allowed']}")
        print(f"   matched [{r['matched_section']}]: {excerpt[:120]}{'...' if len(excerpt) > 120 else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())