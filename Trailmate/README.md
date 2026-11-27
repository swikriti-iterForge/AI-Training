# TrailMate AI

A hiking-trail assistant built in stages: deterministic workflow → vector search → LLM agent →
frontend → observability. **Python only** (FastAPI, LangGraph, pgvector, LiteLLM, Ollama).

## Status

| Stage | What | State |
|---|---|---|
| 0.5 | Environment smoke test | ✅ `scripts/smoke_test.py` |
| 1 | Workflow: LangGraph, rule routing, guardrails, SSE, LiteLLM tiers | ✅ done. Verified with a real model (`fast` and `smart` tiers) |
| 2A | Embedder, pgvector, ingestion, vector search, query validation | ✅ done. Real model, 10/12 on the test queries (baseline) |
| 2B | Hybrid search (keyword + vector, RRF) | built; **not yet measured on the real model**: run `python eval/check_queries.py --compare`. Reranker, chunking comparison and Precision/MRR report: dropped (no problem seen) |
| 3A | Agent mode: search tool, rules-first routing, clarifying question, tie comparison, code-built explanations, difficulty filter, relevance cutoff | built, 199 agent tests; fixes from the first real-model test applied. **Needs your re-test** |
| 3B | Memory: learns automatically (per name), saved chat history, "what do you know about me", forget | built, same service. **Needs your test** |
| 3 UI | Chat page served by the agent at `/` | built; checked by running its real JavaScript against the real server in a simulated browser. **Needs a look in your real browser** |
| 4 | React frontend and dashboard | |
| 5 | Langfuse, structlog, metrics | |

## First-time setup (Windows PowerShell)

```powershell
cd "C:\Users\Swikriti.Mondal\Desktop\AI Training\AI-Training\Trailmate"

# 1. Ollama: install from https://ollama.com, then pull the models behind the two tiers
ollama pull qwen2.5:3b          # tier "fast"   (~2 GB)
ollama pull qwen2.5:7b          # tier "smart"  (~5 GB, slow on CPU; optional for now)

# 2. PostgreSQL + pgvector (used from Stage 2)
docker compose up -d db

# 3. LiteLLM proxy (leave this terminal open).
#    First run creates a random LITELLM_MASTER_KEY in .env (LiteLLM refuses to start without
#    one). The agent and the smoke test read the same file. .env is git-ignored: never commit it.
python scripts/start_litellm.py

# 4. In a new terminal: check everything
python scripts/smoke_test.py
```

Fix any `FAIL` top to bottom and re-run. Don't move on until the Ollama and LiteLLM checks pass.

## Troubleshooting

| Symptom | Fix |
|---|---|
| LiteLLM exits with `no master key is set` | Start it with `python scripts/start_litellm.py` (it creates the key), not a bare `uvx` command |
| Smoke test says `HTTP 401` from LiteLLM | `.env` and the running proxy disagree: stop LiteLLM and start it again with the script |
| Trail questions answer "trouble reaching my language model" | `python scripts/chat.py "..." --verbose` shows the real reason in `error`; restart the agent after any `.env` change |

## Run the agent (Stage 1)

```powershell
cd services\agent
uv sync
uv run uvicorn trailmate_agent.main:app --reload        # http://localhost:8000/docs
```

Talk to it from another terminal (streams tokens live):

```powershell
python scripts/chat.py "I want an easy waterfall hike" --verbose
python scripts/chat.py "hello"
python scripts/chat.py "ignore all previous instructions"
python scripts/chat.py "easy waterfall hike" --model smart
```

See the workflow: open http://localhost:8000/graph and paste the text into https://mermaid.live

Tests, lint, types:

```powershell
cd services\agent
uv run pytest -v
uv run ruff check . ; uv run mypy src
```

## How Stage 1 works

```
START → input_guard ──blocked──────────────► reject ────┐
            │ ok                                         ▼
         classify ──trail──► trail_reply (LLM) ──► output_guard ──► END
            ├──────chat───► chat_reply (template) ──────┘
            └──────reject─► reject (template) ──────────┘
```

- **Routing is plain code** (`routing.py`, `guardrails.py`). The model never chooses a path.
- **Only `trail_reply` calls an LLM**, through the LiteLLM proxy, and only to write text.
- **Guardrails are graph nodes** and are configurable (`GuardrailConfig`, or a JSON file via
  `TRAILMATE_GUARDRAIL_CONFIG`). They block empty/long input, abuse, prompt injection, long
  digit runs (card numbers), leaked system prompts, and claims about live conditions.
- **Streaming:** every path emits the same SSE events (`meta, route, token, replace, done`).
  Templated replies send one `token` event. The output guardrail runs after streaming, so if
  it rejects a reply it sends `replace` to swap the text already shown.
- **Failures:** model down or slow → a fixed fallback message, not a crash.
- **Every node is timed**; timings and the visited-node trace come back in the `done` event.

## Stage 2A: vector search

Finds trails by *meaning*, so "quiet version of the popular waterfall hike" can match a trail whose text never
says "quiet version". Two small services plus a database:

```
POST /search ──► trailmate-search (:8002) ──► trailmate-embedder (:8001)   text → 384 numbers
                      │
                      └─► PostgreSQL + pgvector (:5434)   closest chunks by cosine distance, one per trail
```

**Run it** (four terminals; start from the Trailmate folder, `docker compose up -d db` first):

```powershell
# A. Embedder. The first `uv sync` is large (it installs PyTorch) and the first start downloads a ~90 MB model
cd services\embedder
uv sync
uv run uvicorn trailmate_embedder.main:app --port 8001      # wait for "Application startup complete"

# B. Search: create tables, load the 16 trails, start the service
cd services\search
uv sync
uv run alembic upgrade head
uv run python -m trailmate_search.ingest                    # expect: Ingested 16 trails as 51 chunks
uv run uvicorn trailmate_search.main:app --port 8002

# C. Validate against the 10 provided test queries + the 2 near-duplicate pair queries
python eval/check_queries.py
```

Try it by hand: open http://localhost:8002/docs, or `GET http://localhost:8002/trails/cascade-falls-loop`.
`GET /health` shows how many trails and chunks are loaded.

**How it works**

1. **Chunking (`parsing.py`):** each trail file is split by its `##` headings into chunks. Each chunk is prefixed with
   the trail name and section before embedding ("Cascade Falls Loop - Trail Notes: ..."), so a chunk's vector stays
   tied to *which* trail it belongs to. That is what keeps the near-duplicate trails apart.
2. **Facts vs text:** difficulty, distance, elevation, "best for" and dogs policy are parsed into real columns and
   returned as `facts`. Later, the agent states these from the database, never from the LLM.
3. **Search (`main.py`):** embed the query, compare it with every chunk using cosine distance, keep only the best chunk
   per trail (`DISTINCT ON`), return the top trails with the matching text as evidence.
4. The chunk table has a `strategy` column so Stage 2B can store other chunking strategies side by side.

At 51 chunks an exact scan is instant, so there is no ANN index (HNSW). It would not help here, because the
one-chunk-per-trail rule needs the exact scan. At a much larger scale you would add one and de-duplicate afterwards.

**Troubleshooting**

| Symptom | Fix |
|---|---|
| Embedder fails while downloading the model with a certificate error | Company-proxy certificates again. Tell me the exact error. |
| `uv sync` in `services\embedder` is very slow | Normal the first time: it downloads PyTorch. |
| `alembic upgrade head` can't connect | `docker compose up -d db`, and wait until `docker compose ps` shows `healthy` |
| `ingest` says it cannot reach the embedder | Terminal A isn't running or hasn't finished starting |
| `check_queries.py` says HTTP 503 | The search service can't reach the embedder on port 8001 |

Tests: `uv run pytest -q` inside `services\embedder` (no database needed) and `services\search` (needs the `db` container;
they use a separate `trailmate_search_test` database and skip with a message if Postgres isn't running).

## Stage 3: the agent, with memory (Phase 3)

A second graph beside the Phase 1 workflow. Same message, two modes, so you can compare them:
`--mode workflow` (default, Phase 1: generic text, no trail data) and `--mode agent` (Phase 3).

```
message -> input_guard -> triage --chat / reject--------> templated reply
                            |-- remember / forget / "what do you remember" / start over --> memory_cmd
                            |-- "the second one" --> show_details     |-- "show me more" --> prepare_more --+
                            |-- too vague (and nothing remembered) --> ONE clarifying question              |
                            '-- otherwise --> decide (MODEL: what to search for) --> run_search <-----------+
                                    run_search: filters by difficulty, drops place names, refuses weak matches
                                    tied? -> compare both (code)    else -> explanation built by code from the facts
                            every reply -> output_guard -> END   (also saves what was shown, for "more")
```

**Who decides what.** Code: the obvious cases, whether a request is too vague, difficulty and features read from
your words, whether results are tied, whether a match is too weak, the facts, and the explanation. The model:
what to search for. The model-written recommendation still exists (`TRAILMATE_EXPLAIN_WITH_LLM=true`) but is off.

**What the first real-model test changed.**

| Seen in the test | Fix |
|---|---|
| The model said a trail was "near Mumbai" (the data has no locations) | The explanation is built by code, so it can't echo it. Place names are also removed from the search (they lowered the top score from 0.640 to 0.529) |
| The model credited Blackrock Summit with a waterfall it doesn't have | "Why it fits / Not matched" is computed from what each trail's text really contains (`terms_found` from the search service) |
| A Hard trail ranked 2nd for "easy"; an easy one ranked 1st for "hard scramble" | Difficulty is read from the request and applied as a filter. If no trail has that rating, the filter is dropped and you are told |
| "Somewhere to clear my head" scored 0.14 and still got a confident answer | A relevance cutoff (`TRAILMATE_MIN_RELEVANCE`, default 0.25, **provisional**): weak matches get a question instead of a recommendation |
| Writing the answer took 26-35 s | No second model call: one call to decide, then code |
| "okay show some easy ones" after an alpine search was refused ("couldn't find a match"): the remembered "alpine, snow" were added to the new search, and a short request like "easy ones" scores low whatever it matches | A request that is only a filter (how hard, no subject) is never refused for a low score. Remembered features that make the search match nothing, or that no result mentions, are left out, and the reply says so |
| "hi, i want short and long" and "short and challenging" were refused: "short" and "long" were treated as search words, but they describe how long a trail is | "Short" and "long" are real filters on each trail's distance (`short` = up to 4 km, `long` = 8 km or more: every Easy trail is <= 4 km and every Hard trail >= 8.2 km in this data; `TRAILMATE_SHORT_MAX_KM` / `TRAILMATE_LONG_MIN_KM`). "Short and long" asks what you want; if nothing fits every filter it loosens one at a time, the one carried over from earlier first, and says so |

**Search modes** (`TRAILMATE_SEARCH_MODE`, default `vector`): `vector` (meaning), `keyword` (PostgreSQL full-text,
with stemming), `hybrid` (both, fused with Reciprocal Rank Fusion). `score` is always the cosine similarity, so
the cutoff and tie checks mean the same thing in every mode. Don't switch the default until
`python eval/check_queries.py --compare` shows hybrid is clearly better on the real model.

**Memory.** It learns automatically, per name, and you stay in control.

| | Short-term | Long-term |
|---|---|---|
| What | the conversation: difficulty and features so far, which trails were shown | what it learned about you |
| How | automatic | automatic: every search request is counted; asked for at least 2 times (`TRAILMATE_MIN_MENTIONS`) means a **usual preference**; a standing statement ("I love waterfalls", "remember that I prefer easy trails") counts at once. A one-off request is counted, never assumed |
| Where | LangGraph checkpointer, one thread per user and session (RAM). After a restart it is restored from the saved chat | `data/memory/preferences.json`, one entry per name |
| Use | "show me more" (skips what was shown), "the second one", "something harder" (keeps your waterfall) | applied when a request doesn't say otherwise, and the reply says so: "(Using what I remember about you: easy trails.)" |
| Control | `start over` | ask "what do you know about me" / "do you know what trails I like"; `forget my preferences`; the **What I remember** box; the **Remember me** switch (off = nothing is saved or learned); `TRAILMATE_LEARN_AUTOMATICALLY=false`; `GET`/`DELETE /memory/{user_id}` |

A new topic replaces the old features, a refinement keeps them. Difficulty carries over until you change it.

**Chat history.** With a name set (and Remember me on), every conversation is saved to
`data/memory/chats/<name>/<chat>.json`: the messages, the trail cards, and the session memory. Reopen one from the
sidebar and carry on, even after a restart ("show me more" still knows what you saw). API:
`GET /conversations/{user}`, `GET`/`DELETE /conversations/{user}/{chat}`, `DELETE /conversations/{user}`.
Everything is plain files you can open and delete. **Not authenticated**: anyone who knows a name can read that name's
chats and memory. Fine locally, not for production. Names cannot contain slashes or start with a dot, so they can never
point outside the data folder.

**Run it** (four terminals; start from the Trailmate folder, `docker compose up -d db` first):

```powershell
# A. Embedder. The first `uv sync` is large (it installs PyTorch) and the first start downloads a ~90 MB model
cd services\embedder
uv sync
uv run uvicorn trailmate_embedder.main:app --port 8001      # wait for "Application startup complete"

# B. Search: create tables, load the 16 trails, start the service
cd services\search
uv sync
uv run alembic upgrade head
uv run python -m trailmate_search.ingest                    # expect: Ingested 16 trails as 51 chunks
uv run uvicorn trailmate_search.main:app --port 8002

# C. Validate against the 10 provided test queries + the 2 near-duplicate pair queries
python eval/check_queries.py
```

Try it by hand: open http://localhost:8002/docs, or `GET http://localhost:8002/trails/cascade-falls-loop`.
`GET /health` shows how many trails and chunks are loaded.

**How it works**

1. **Chunking (`parsing.py`):** each trail file is split by its `##` headings into chunks. Each chunk is prefixed with
   the trail name and section before embedding ("Cascade Falls Loop - Trail Notes: ..."), so a chunk's vector stays
   tied to *which* trail it belongs to. That is what keeps the near-duplicate trails apart.
2. **Facts vs text:** difficulty, distance, elevation, "best for" and dogs policy are parsed into real columns and
   returned as `facts`. Later, the agent states these from the database, never from the LLM.
3. **Search (`main.py`):** embed the query, compare it with every chunk using cosine distance, keep only the best chunk
   per trail (`DISTINCT ON`), return the top trails with the matching text as evidence.
4. The chunk table has a `strategy` column so Stage 2B can store other chunking strategies side by side.

At 51 chunks an exact scan is instant, so there is no ANN index (HNSW). It would not help here, because the
one-chunk-per-trail rule needs the exact scan. At a much larger scale you would add one and de-duplicate afterwards.

**Troubleshooting**

| Symptom | Fix |
|---|---|
| Embedder fails while downloading the model with a certificate error | Company-proxy certificates again. Tell me the exact error. |
| `uv sync` in `services\embedder` is very slow | Normal the first time: it downloads PyTorch. |
| `alembic upgrade head` can't connect | `docker compose up -d db`, and wait until `docker compose ps` shows `healthy` |
| `ingest` says it cannot reach the embedder | Terminal A isn't running or hasn't finished starting |
| `check_queries.py` says HTTP 503 | The search service can't reach the embedder on port 8001 |

Tests: `uv run pytest -q` inside `services\embedder` (no database needed) and `services\search` (needs the `db` container;
they use a separate `trailmate_search_test` database and skip with a message if Postgres isn't running).

## Stage 3: the agent, with memory (Phase 3)

A second graph beside the Phase 1 workflow. Same message, two modes, so you can compare them:
`--mode workflow` (default, Phase 1: generic text, no trail data) and `--mode agent` (Phase 3).

```
message -> input_guard -> triage --chat / reject--------> templated reply
                            |-- remember / forget / "what do you remember" / start over --> memory_cmd
                            |-- "the second one" --> show_details     |-- "show me more" --> prepare_more --+
                            |-- too vague (and nothing remembered) --> ONE clarifying question              |
                            '-- otherwise --> decide (MODEL: what to search for) --> run_search <-----------+
                                    run_search: filters by difficulty, drops place names, refuses weak matches
                                    tied? -> compare both (code)    else -> explanation built by code from the facts
                            every reply -> output_guard -> END   (also saves what was shown, for "more")
```

**Who decides what.** Code: the obvious cases, whether a request is too vague, difficulty and features read from
your words, whether results are tied, whether a match is too weak, the facts, and the explanation. The model:
what to search for. The model-written recommendation still exists (`TRAILMATE_EXPLAIN_WITH_LLM=true`) but is off.

**What the first real-model test changed.**

| Seen in the test | Fix |
|---|---|
| The model said a trail was "near Mumbai" (the data has no locations) | The explanation is built by code, so it can't echo it. Place names are also removed from the search (they lowered the top score from 0.640 to 0.529) |
| The model credited Blackrock Summit with a waterfall it doesn't have | "Why it fits / Not matched" is computed from what each trail's text really contains (`terms_found` from the search service) |
| A Hard trail ranked 2nd for "easy"; an easy one ranked 1st for "hard scramble" | Difficulty is read from the request and applied as a filter. If no trail has that rating, the filter is dropped and you are told |
| "Somewhere to clear my head" scored 0.14 and still got a confident answer | A relevance cutoff (`TRAILMATE_MIN_RELEVANCE`, default 0.25, **provisional**): weak matches get a question instead of a recommendation |
| Writing the answer took 26-35 s | No second model call: one call to decide, then code |

**Search modes** (`TRAILMATE_SEARCH_MODE`, default `vector`): `vector` (meaning), `keyword` (PostgreSQL full-text,
with stemming), `hybrid` (both, fused with Reciprocal Rank Fusion). `score` is always the cosine similarity, so
the cutoff and tie checks mean the same thing in every mode. Don't switch the default until
`python eval/check_queries.py --compare` shows hybrid is clearly better on the real model.

**Memory.**

| | Short-term | Long-term |
|---|---|---|
| What | the conversation: difficulty and features so far, which trails were shown | only what you ask it to keep |
| Where | LangGraph checkpointer, one thread per `session_id` (RAM: lost on restart) | `data/memory/preferences.json`, one entry per `user_id` |
| Use | "show me more" (skips what was shown), "the second one", "something harder" (keeps your waterfall) | applied automatically in new chats, and the reply says so |
| Control | `start over` | `remember that I prefer easy trails`, `what do you remember about me`, `forget my preferences`, `GET/DELETE /memory/{user_id}` |

A new topic replaces the old features, a refinement keeps them. Difficulty carries over until you change it.
Not authenticated: anyone who knows a user id can read that user's preferences. Fine locally, not for production.

**Run it** (many terminals; Stage 3 needs everything):

```powershell
docker compose up -d db
# 1. embedder   (services\embedder)  uv run uvicorn trailmate_embedder.main:app --port 8001
# 2. search     (services\search)    uv run alembic upgrade head   (once; adds the keyword column)
#                                    uv run uvicorn trailmate_search.main:app --port 8002
# 3. LiteLLM    (Trailmate)          python scripts/start_litellm.py
# 4. agent      (services\agent)     uv run uvicorn trailmate_agent.main:app
# 5. talk       open http://localhost:8000/   (the chat UI)     or     python scripts/chat.py -i --verbose
```

**Measure instead of guessing** (embedder and search running):

```powershell
python eval/check_queries.py --compare     # vector vs keyword vs hybrid on the 12 test queries
python eval/check_cutoff.py                # where should the "no good match" cutoff sit?
```

## Chat UI

Start the agent and open **http://localhost:8000/** (use your port if it differs). It is one HTML file served by the
agent itself, so there is nothing to install or build, and it needs no internet.

- Type a message or click a suggestion. The answer **streams in as it is written**; while the agent works you see
  "Searching hiking trails..." and, on the first request after idle, a note that the model is waking up (about 20 s).
- Trails appear as **cards** (difficulty, distance, climb, best for, dogs) with facts straight from the database.
  **Tell me more** asks for that trail; **Show me more** asks for the next ones. Buttons on older results switch
  themselves off, because "the second one" always means the latest list.
- **Sidebar:** your saved chats (click to reopen, x to delete: it asks twice) and **What I remember** (what it knows
  about you, with a **Forget** button). On a phone it opens with the **History** button.
- **Header:** Mode (Agent, or the Phase 1 Workflow for comparison), Model tier, **Name** (chats and memory are saved
  under it), **Remember me** (untick to chat without saving or learning anything), Debug, New chat.
- A refresh reopens the chat the tab was in. Errors say what happened and offer **Try again**.
- Text from the model or from you is always inserted as plain text, never as HTML.

The UI talks only to this server (`/chat`, `/health`, `/conversations`, `/memory`). Page tests: `services/agent/tests/test_ui.py`.

## Dataset (`data/`)

The provided sample set: 16 trail write-ups (`trails/*.md`) and `test_queries.md` (10 queries with
expected results plus 2 near-duplicate pairs). Used as-is. Notes that shape the design:

- Each trail has five structured facts: Difficulty, Distance, Elevation Gain, Best For, Dogs Allowed.
  The agent will quote these from parsed data, never from LLM memory.
- **There is no location field.** "Near Mumbai" can't be answered from this data, so the agent
  will say that instead of guessing.
- Not live data: nothing here says anything about current conditions or weather.
