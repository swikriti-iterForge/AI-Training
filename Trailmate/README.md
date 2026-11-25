# TrailMate AI

A hiking-trail assistant built in stages: deterministic workflow → vector search → LLM agent →
frontend → observability. **Python only** (FastAPI, LangGraph, pgvector, LiteLLM, Ollama).

## Status

| Stage | What | State |
|---|---|---|
| 0.5 | Environment smoke test | ✅ `scripts/smoke_test.py` |
| 1 | Workflow: LangGraph, rule routing, guardrails, SSE, LiteLLM tiers | ✅ done. Verified with a real model (`fast` and `smart` tiers) |
| 2A | Embedder, pgvector, ingestion, vector search, query validation | ✅ done. Real model, 10/12 on the test queries (baseline) |
| 2B | Hybrid search (keyword + RRF), chunking experiments, evaluation harness | **deferred until after Phase 3.** Baseline: 10/12; targets are query #3 (hard scramble + waterfall) and #9 (Extension must outrank Loop) |
| 3 | Agent, tools, memory, explainable recommendations | **next.** Step 1: `scripts/tool_call_check.py` (can the local models call a tool correctly?) |
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

## Dataset (`data/`)

The provided sample set: 16 trail write-ups (`trails/*.md`) and `test_queries.md` (10 queries with
expected results plus 2 near-duplicate pairs). Used as-is. Notes that shape the design:

- Each trail has five structured facts: Difficulty, Distance, Elevation Gain, Best For, Dogs Allowed.
  The agent will quote these from parsed data, never from LLM memory.
- **There is no location field.** "Near Mumbai" can't be answered from this data, so the agent
  will say that instead of guessing.
- Not live data: nothing here says anything about current conditions or weather.
