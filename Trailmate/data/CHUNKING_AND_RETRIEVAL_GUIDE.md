# Chunking & Retrieval Guide — Sample Trail Dataset

This is optional background, not a spec. The plan asks you to design your own chunking and retrieval
approach — use this only if you want a sanity check on your reasoning, or a starting point if you're
genuinely stuck.

This dataset is 16 short trail write-ups (`trails/*.md`), deliberately varied so a simple chunking and
retrieval approach can be exercised end to end without needing anything sophisticated.

## What's in the dataset, and why

- **14 "normal" trails** — each with the same three sections: Overview, Details, Trail Notes.
- **One short doc** — `meadowbrook-path.md` — to check your chunker doesn't do something strange with
  a document that barely fills one chunk.
- **One long, multi-section doc** — `thunder-gorge-trail.md` — with extra headings (Water Crossings,
  Safety Notes, Best Season) to check your chunker handles a document with more structure than the rest.
- **Two near-duplicate pairs** — see `test_queries.md` — to check your retrieval can tell closely related
  trails apart, not just find "a waterfall trail" in general.

## Suggested chunking strategy (keep it simple)

- **Split by heading section**, not by a fixed character count. These documents are short enough that
  heading-based splitting (Overview / Details / Trail Notes / etc.) alone works well — you don't need a
  recursive character splitter or token-counting logic for a dataset this size.
- **Prefix each chunk with the trail name and section heading** before embedding, e.g.:
  `"Cascade Falls Loop — Trail Notes: The trail is wide and well-maintained..."`
  This keeps a chunk's embedding anchored to which trail and section it came from, even once it's just a
  vector sitting in a table next to a dozen others.
- **Don't bother with chunk overlap here.** Overlap exists to avoid cutting a relevant sentence in half at
  a chunk boundary on long documents — these sections are short enough that overlap would mostly just
  duplicate content rather than add value. It's worth knowing about for later, larger documents, but skip
  it for this dataset.

## Suggested retrieval strategy (keep it simple)

- **Embed every chunk once at ingest time**, storing at minimum: `trail_id`, `section_name`, `chunk_text`,
  and the `embedding` vector.
- **On a query:** embed the query text the same way you embedded the chunks, run a cosine similarity
  search, and take the top-k chunks (k=5 is a reasonable starting point).
- **De-duplicate by trail before returning results.** Multiple chunks will often come from the same trail
  (e.g. both the Details and Trail Notes sections of Thunder Gorge Trail might match a query about
  waterfalls) — keep only the highest-scoring chunk per trail, then return the top 3 distinct trails to
  the agent, not the top 3 chunks.

## Validating it worked

Run the queries in `test_queries.md` against your index once everything is loaded. You're checking for
"is the top result thematically sensible," not an exact match — several queries have more than one
reasonable answer. The two near-duplicate pairs are the more interesting check: if your retrieval can't
tell Cascade Falls Loop and Cascade Falls Extension apart on a query that should favor one over the
other, that's a sign your chunks are losing distinguishing detail somewhere in the pipeline.
