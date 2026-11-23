# Retrieval Test Queries

Use these to sanity-check your retrieval once trails are indexed. For each query, the top 1-2 results
should be thematically sensible. Several queries have more than one reasonable answer on purpose —
the point is relevance, not a single "correct" match.

| # | Query | Expect to see near the top |
|---|-------|------------------------------|
| 1 | "I want an easy trail near water for tomorrow morning" | Silver Lake Circuit, Hidden Spring Trail, Cascade Falls Loop |
| 2 | "Something short and dog-friendly, totally flat" | Willow Creek Wander, Old Mill Loop |
| 3 | "Looking for a hard scramble with a waterfall" | Thunder Gorge Trail |
| 4 | "Best sunset viewpoint hike" | Sunset Bluff Trail |
| 5 | "Family hike with wildflowers, nothing too long" | Meadowbrook Path |
| 6 | "Alpine trail where I might see snow" | Timberline Pass |
| 7 | "Coastal hike with some exposure" | Rocky Point Trail |
| 8 | "Quiet forest trail with a creek crossing" | Fern Valley Loop |
| 9 | "A waterfall hike but not too crowded" | Cascade Falls Extension (over the shorter, busier Cascade Falls Loop) |
| 10 | "Steep climb with a big payoff view, I don't mind exposure" | Eagle's Perch Trail, Blackrock Summit |

## Two deliberate near-duplicate pairs

These exist specifically to test whether your retrieval can tell closely related trails apart, not just
find "a waterfall trail" or "a dog-friendly trail" in general:

- **Cascade Falls Loop** vs. **Cascade Falls Extension** — same trailhead and first two kilometers,
  different length and crowd level. A query like "quiet version of the popular waterfall hike" should
  favor the Extension.
- **Willow Creek Wander** vs. **Old Mill Loop** — both flat, easy, and dog-friendly. A query mentioning
  "history" or "old ruins" should favor Old Mill Loop specifically, not just return either one at random.

If your retrieval can't distinguish these pairs on the queries that should separate them, that's a good
signal your chunks are losing distinguishing detail — check whether section-level context (which trail,
which section) is actually making it into what you embed.
