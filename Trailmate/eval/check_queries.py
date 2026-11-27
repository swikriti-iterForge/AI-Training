"""Validate search quality against data/queries.json: the 10 provided test queries plus a near-duplicate
pair query for each deliberate pair (Cascade Falls Loop vs Extension, Willow Creek Wander vs Old Mill Loop).

    python eval/check_queries.py                    # the service's default mode, full detail
    python eval/check_queries.py --mode hybrid      # vector | keyword | hybrid
    python eval/check_queries.py --compare          # all three modes side by side

The provided test file says the top 1-2 results should be sensible, so the rules are:
  * normal query : PASS if an expected trail is in the TOP 2
  * pair query   : PASS only if the right trail ranks ABOVE its near-duplicate (query #9 and #11, #12)
Standard library only.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS_RANK = 2  # the provided test file: "the top 1-2 results should be sensible"
MODES = ["vector", "keyword", "hybrid"]


class ServiceError(Exception):
    pass


def search(url: str, query: str, top_k: int, mode: str | None) -> dict:
    payload: dict = {"query": query, "top_k": top_k}
    if mode:
        payload["mode"] = mode
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:  # reached the service, but it reported a failure
        hint = " It cannot reach the embedder: is trailmate-embedder running on port 8001?" if exc.code == 503 else ""
        raise ServiceError(f"The search service answered HTTP {exc.code}. Check its terminal.{hint}") from exc
    except (urllib.error.URLError, OSError) as exc:  # could not connect at all
        raise ServiceError(f"Cannot reach the search service at {url}: {exc}\nStart the services first (see README).") from exc


def judge(q: dict, data: dict) -> tuple[bool, int | None, str]:
    """-> (passed, rank of the first expected trail, what was expected)."""
    slugs = [r["slug"] for r in data["results"]]
    rank = {slug: i + 1 for i, slug in enumerate(slugs)}
    first = min((rank[s] for s in q["relevant"] if s in rank), default=None)
    if "must_outrank" in q:
        better, worse = rank.get(q["relevant"][0]), rank.get(q["must_outrank"])
        ok = better is not None and (worse is None or better < worse)
        return ok, first, f"{q['relevant'][0]} above {q['must_outrank']}"
    return first is not None and first <= PASS_RANK, first, " or ".join(q["relevant"])


def run_mode(url: str, queries: list[dict], mode: str | None, show: bool) -> dict:
    passed = pairs_passed = pairs_total = 0
    first_ranks: list[int | None] = []
    verdicts: dict[int, tuple[bool, int | None]] = {}
    latencies: list[float] = []
    for q in queries:
        data = search(url, q["query"], 5, mode)  # 5, so a near-duplicate competitor is visible
        ok, first, expectation = judge(q, data)
        latencies.append(data["timing_ms"]["total"])
        first_ranks.append(first)
        verdicts[q["id"]] = (ok, first)
        passed += ok
        if "must_outrank" in q:
            pairs_total += 1
            pairs_passed += ok
        if show:
            found = f"first expected trail at rank {first}" if first else "expected trail not in the top 5"
            print(f"\n#{q['id']:<2} {'PASS' if ok else 'FAIL'}  {q['query']}")
            print(f"     expected: {expectation}   ({found})")
            for i, r in enumerate(data["results"][:3], 1):
                mark = "*" if r["slug"] in q["relevant"] else " "
                print(f"     {i}.{mark} {r['name']:<26} score {r['score']:.3f}  ({r['matched_section']})")
    n = len(queries)
    at = lambda k: sum(1 for r in first_ranks if r is not None and r <= k)  # noqa: E731
    return {"passed": passed, "n": n, "pairs": (pairs_passed, pairs_total), "at": (at(1), at(2), at(3)),
            "latency": (sum(latencies) / n, max(latencies)), "verdicts": verdicts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8002/search")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.json")
    parser.add_argument("--mode", choices=MODES, help="search mode (default: whatever the service is set to)")
    parser.add_argument("--compare", action="store_true", help="run vector, keyword and hybrid and compare")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe when output is redirected to a file
    queries = json.loads(args.queries.read_text(encoding="utf-8"))

    try:
        if args.compare:
            results = {m: run_mode(args.url, queries, m, show=False) for m in MODES}
            print(f"{'#':<4}{'query':<58}" + "".join(f"{m:>10}" for m in MODES))
            for q in queries:
                cells = []
                for m in MODES:
                    ok, first = results[m]["verdicts"][q["id"]]
                    cells.append(f"{'PASS' if ok else 'FAIL'} r{first if first else '-'}")
                print(f"{q['id']:<4}{q['query'][:56]:<58}" + "".join(f"{c:>10}" for c in cells))
            print(f"\n{'-' * 84}\n{'':<62}" + "".join(f"{m:>10}" for m in MODES))
            rows = [
                ("Passed", lambda r: f"{r['passed']}/{r['n']}"),
                ("Near-duplicate pairs", lambda r: f"{r['pairs'][0]}/{r['pairs'][1]}"),
                ("Expected trail at rank 1", lambda r: f"{r['at'][0]}/{r['n']}"),
                ("  within top 2", lambda r: f"{r['at'][1]}/{r['n']}"),
                ("  within top 3", lambda r: f"{r['at'][2]}/{r['n']}"),
                ("Mean latency (ms)", lambda r: f"{r['latency'][0]:.0f}"),
            ]
            for label, fmt in rows:
                print(f"{label:<62}" + "".join(f"{fmt(results[m]):>10}" for m in MODES))
            print("\nr = rank of the first expected trail ('-' = not in the top 5). 12 queries is a small sample:")
            print("a one-query difference is noise. Switch the service default only if a mode is clearly better.")
            return 0

        r = run_mode(args.url, queries, args.mode, show=True)
    except ServiceError as exc:
        print(exc)
        return 2
    print(f"\n{'-' * 64}")
    print(f"Passed {r['passed']}/{r['n']}   (normal: expected trail in top {PASS_RANK}; pair: right trail above its near-duplicate)")
    print(f"Near-duplicate pairs: {r['pairs'][0]}/{r['pairs'][1]}")
    print(f"First expected trail at rank 1: {r['at'][0]}/{r['n']} | within top 2: {r['at'][1]}/{r['n']} | within top 3: {r['at'][2]}/{r['n']}")
    print(f"Search latency (embed + search): mean {r['latency'][0]:.0f} ms, max {r['latency'][1]:.0f} ms")
    print("* = an expected trail")
    return 0 if r["passed"] == r["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
