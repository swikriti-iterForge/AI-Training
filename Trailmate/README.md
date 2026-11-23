# TrailMate AI

A hiking-trail assistant built in stages: deterministic workflow → vector search → LLM agent →
frontend → observability. **Python only** (FastAPI, LangGraph, pgvector, LiteLLM, Ollama).

## Status

| Stage | What | State |
|---|---|---|
| 0.5 | Environment smoke test | ✅ `scripts/smoke_test.py` |
| 1 | Workflow: LangGraph, rule routing, guardrails, SSE, LiteLLM tiers | ✅ built, 61 tests pass; **not yet run against a real model** |
| 2 | Embedder, pgvector, hybrid search, evaluation | next |
| 3 | Agent, tools, memory, explainable recommendations | |
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

## Dataset (`data/`)

The provided sample set: 16 trail write-ups (`trails/*.md`) and `test_queries.md` (10 queries with
expected results plus 2 near-duplicate pairs). Used as-is. Notes that shape the design:

- Each trail has five structured facts: Difficulty, Distance, Elevation Gain, Best For, Dogs Allowed.
  The agent will quote these from parsed data, never from LLM memory.
- **There is no location field.** "Near Mumbai" can't be answered from this data, so the agent
  will say that instead of guessing.
- Not live data: nothing here says anything about current conditions or weather.
