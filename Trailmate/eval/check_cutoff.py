"""Where should the 'no good match' cutoff sit?

The agent refuses to recommend when the best similarity score is below a cutoff (TRAILMATE_MIN_RELEVANCE).
A cutoff is only sensible if real questions score clearly higher than out-of-scope ones. This script
measures both groups so the cutoff is chosen from data, not guessed:

    python eval/check_cutoff.py                  # needs the embedder and search service running
    python eval/check_cutoff.py --mode hybrid

Positive questions: data/queries.json (each has a good answer). Negative questions: data/negative_queries.json
(no trail is a good answer). The score is the best cosine similarity among the top results. Standard library only.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_queries import ServiceError, search  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def top_score(url: str, query: str, mode: str | None) -> float:
    results = search(url, query, 3, mode)["results"]
    return max((r["score"] for r in results), default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8002/search")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    positives = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))
    negatives = json.loads((ROOT / "data" / "negative_queries.json").read_text(encoding="utf-8"))
    try:
        pos = [(q["query"], top_score(args.url, q["query"], args.mode)) for q in positives]
        neg = [(q["query"], top_score(args.url, q["query"], args.mode)) for q in negatives]
    except ServiceError as exc:
        print(exc)
        return 2

    for title, rows in (("Real questions (a good answer exists)", pos), ("Out-of-scope questions (no good answer)", neg)):
        print(f"\n{title}")
        for query, score in sorted(rows, key=lambda r: r[1]):
            print(f"  {score:.3f}  {query}")

    lowest_pos, highest_neg = min(s for _, s in pos), max(s for _, s in neg)
    print(f"\nLowest real-question score : {lowest_pos:.3f}")
    print(f"Highest out-of-scope score : {highest_neg:.3f}")
    if lowest_pos > highest_neg:
        print(f"The groups do not overlap. Any cutoff between {highest_neg:.3f} and {lowest_pos:.3f} separates them.")
        print(f"A cautious choice is {highest_neg + (lowest_pos - highest_neg) / 3:.2f} (closer to the out-of-scope side, so real questions are rarely refused).")
        print("Set it with the TRAILMATE_MIN_RELEVANCE environment variable (or in .env) and restart the agent.")
    else:
        print("The groups OVERLAP: some out-of-scope questions score as high as real ones.")
        print("No single cutoff separates them cleanly. Choose one that favours refusing weak matches,")
        print("and expect some real questions to be asked to rephrase.")
    print(f"\nSample size: {len(pos)} real and {len(neg)} out-of-scope questions. Treat the result as a first estimate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
