"""Validate search quality against data/queries.json: the 10 provided test queries plus a near-duplicate
pair query for each deliberate pair (Cascade Falls Loop vs Extension, Willow Creek Wander vs Old Mill Loop).

    python eval/check_queries.py                  # needs the embedder and search service running

The provided test file says the top 1-2 results should be sensible, so the rules are:
  * normal query : PASS if an expected trail is in the TOP 2
  * pair query   : PASS only if the right trail ranks ABOVE its near-duplicate (query #9 and #11, #12)
The summary also shows how often the first expected trail was rank 1, within the top 2, and within the top 3.
Standard library only.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS_RANK = 2  # the provided test file: "the top 1-2 results should be thematically sensible"


def search(url: str, query: str, top_k: int) -> dict:
    request = urllib.request.Request(
        url, json.dumps({"query": query, "top_k": top_k}).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8002/search")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe when output is redirected to a file

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    passed = pairs_passed = pairs_total = 0
    first_ranks: list[int | None] = []
    latencies: list[float] = []

    for q in queries:
        try:
            data = search(args.url, q["query"], top_k=5)  # 5, so a near-duplicate competitor is visible
        except urllib.error.HTTPError as exc:  # reached the service, but it reported a failure
            print(f"The search service answered HTTP {exc.code} for query #{q['id']}. Check its terminal for the error.")
            if exc.code == 503:
                print("503 means it cannot reach the embedder: is trailmate-embedder running on port 8001?")
            return 2
        except (urllib.error.URLError, OSError) as exc:  # could not connect at all
            print(f"Cannot reach the search service at {args.url}: {exc}\nStart the services first (see README).")
            return 2
        slugs = [r["slug"] for r in data["results"]]
        rank = {slug: i + 1 for i, slug in enumerate(slugs)}
        latencies.append(data["timing_ms"]["total"])
        first = min((rank[s] for s in q["relevant"] if s in rank), default=None)
        first_ranks.append(first)

        if "must_outrank" in q:
            pairs_total += 1
            better, worse = rank.get(q["relevant"][0]), rank.get(q["must_outrank"])
            ok = better is not None and (worse is None or better < worse)
            pairs_passed += ok
            expectation = f"{q['relevant'][0]} above {q['must_outrank']}"
        else:
            ok = first is not None and first <= PASS_RANK
            expectation = " or ".join(q["relevant"])
        passed += ok

        found = f"first expected trail at rank {first}" if first else "expected trail not in the top 5"
        print(f"\n#{q['id']:<2} {'PASS' if ok else 'FAIL'}  {q['query']}")
        print(f"     expected: {expectation}   ({found})")
        for i, r in enumerate(data["results"][:3], 1):
            mark = "*" if r["slug"] in q["relevant"] else " "
            print(f"     {i}.{mark} {r['name']:<26} score {r['score']:.3f}  ({r['matched_section']})")

    n = len(queries)
    at = lambda k: sum(1 for r in first_ranks if r is not None and r <= k)  # noqa: E731
    print(f"\n{'-' * 64}")
    print(f"Passed {passed}/{n}   (normal: expected trail in top {PASS_RANK}; pair: right trail above its near-duplicate)")
    print(f"Near-duplicate pairs: {pairs_passed}/{pairs_total}")
    print(f"First expected trail at rank 1: {at(1)}/{n} | within top 2: {at(2)}/{n} | within top 3: {at(3)}/{n}")
    print(f"Search latency (embed + search): mean {sum(latencies) / n:.0f} ms, max {max(latencies):.0f} ms")
    print("* = an expected trail")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
