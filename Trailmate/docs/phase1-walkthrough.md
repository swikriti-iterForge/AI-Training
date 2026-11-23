# TrailMate Phase 1: Workflow Orchestration with LangGraph

**Walkthrough, demo script and results**

Author: Swikriti Mondal · Scope: Phase 1 of the IterForge training program

---

## 1. What Phase 1 delivers

A small conversational service where **every routing decision is made by ordinary code, not by a model**. The
language model is trusted only to write text on one path. The structure of the graph, not prompt
engineering, is what makes behaviour predictable.

| Capability | How it is done |
|---|---|
| Orchestration | LangGraph `StateGraph` with 6 nodes and conditional edges |
| Routing | Rule-based classifier (`routing.py`), no model involved |
| Model access | LiteLLM proxy with named tiers (`fast`, `smart`) in front of Ollama |
| Streaming | Server-Sent Events (SSE) via `sse-starlette`, same event sequence on every path |
| Safety | Input and output guardrails as graph nodes, configurable and tested |
| Reliability | Model failure returns a fixed fallback message instead of an error |
| Observability (basic) | Every node is timed; the visited-node trace is returned with each reply |

## 2. The graph

```mermaid
flowchart TD
    S(["POST /chat"]) --> IG[input_guard]
    IG -- blocked --> RJ[reject<br/>templated]
    IG -- ok --> CL[classify<br/>plain rules]
    CL -- trail --> TR[trail_reply<br/>LLM via LiteLLM]
    CL -- chat --> CH[chat_reply<br/>templated]
    CL -- reject --> RJ
    TR --> OG[output_guard]
    CH --> OG
    RJ --> OG
    OG --> E(["SSE stream: meta, route, token..., done"])
```

The live diagram is always available at `GET /graph` (paste it into https://mermaid.live).

**Two kinds of "no":** the input guardrail stops messages that are *unsafe* (injection, abuse, empty, too long,
card-like numbers). The classifier turns away requests the app *cannot serve* (live weather, off-topic). Both
end at the same `reject` node, but a blocked message skips `classify` entirely.

## 3. Where things live

| File (`services/agent/src/trailmate_agent/`) | Role | Spring Boot analogy |
|---|---|---|
| `main.py` | FastAPI app, `/chat` SSE endpoint, `/graph`, `/health` | `@RestController` |
| `graph.py` | The LangGraph workflow and node functions | the service layer / orchestration |
| `routing.py` | Keyword classifier and templated replies | a rules engine |
| `guardrails.py` | Input/output checks, `GuardrailConfig` | validation filters |
| `llm.py` | `LLMClient` protocol and the LiteLLM client | an interface with one implementation |
| `config.py` | Settings from environment and `.env` | `@ConfigurationProperties` |

Supporting files: `infra/litellm_config.yaml` (tier → model mapping), `scripts/start_litellm.py`,
`scripts/smoke_test.py`, `scripts/chat.py` (streaming terminal client).

## 4. SSE event contract

Every path emits the same sequence, so a client needs one code path.

| Event | Payload | Meaning |
|---|---|---|
| `meta` | `request_id`, `tier` | first event |
| `route` | `path`, `reason` | `trail`, `chat`, `reject` or `blocked` |
| `token` | `text` | a piece of the reply. Templated replies send **one** token event with the full text |
| `replace` | `text` | a guardrail or failure swapped text that was already streamed |
| `done` | `category`, `reply_source`, `trace`, `timings_ms`, `total_ms`, `error` | final summary |
| `error` | `message` | only on an unexpected crash |

`reply_source` is `template`, `llm` or `fallback`.

---

## 5. Demo script (about 7 minutes)

### Before starting
Three terminals, all from the `Trailmate` folder unless noted.

```text
Terminal 1:  python scripts/start_litellm.py          # leave running
Terminal 2:  cd services/agent
             uv run uvicorn trailmate_agent.main:app  # leave running
Terminal 3:  the one you demo from
```

Warm the model first, or the first request pays the model load time (about 6 seconds measured):

```text
python scripts/chat.py "easy hike please"
```

### Step 1: show the graph (1 min)
Open `http://localhost:8000/graph` and paste the text into mermaid.live, or show the diagram in section 2.

> *"Every box is a node. The diagonal arrows are conditional edges, and the code that picks them is plain
> Python. The model only appears in one box, and only to write text."*

### Step 2: chat path, templated (30 s)
```text
python scripts/chat.py "hello" --verbose
```
Point at: `route: chat`, `reply_source: template`, and the trace
`input_guard → classify → chat_reply → output_guard`.

> *"No model was called. This reply is a fixed template, so it is instant and always the same."*

### Step 3: reject path (30 s)
```text
python scripts/chat.py "what's the weather on the trail today" --verbose
```
Point at: `route: reject`, trace includes `classify`.

> *"I have no live data, so the app refuses instead of guessing. That was a deliberate rule."*

### Step 4: blocked by the guardrail (45 s)
```text
python scripts/chat.py "ignore all previous instructions" --verbose
```
Point at: `route: blocked`, `block_reason: prompt_injection`, and that **`classify` is missing from the trace**.

> *"The guardrail stopped it before classification. A hijack attempt costs nothing and never reaches the model."*

### Step 5: the model path, streaming (1.5 min)
```text
python scripts/chat.py "I want an easy waterfall hike" --verbose
```
Point at: words appearing gradually, `reply_source: llm`, and in `timings_ms` that `trail_reply` is almost the
entire request.

> *"The model is the only slow part. The other three nodes together take a few milliseconds."*

### Step 6: switch model with no code change (45 s)
```text
python scripts/chat.py "I want an easy waterfall hike" --model smart --verbose
```
> *"The app asks for a tier name. Which model sits behind `smart` is one line in `litellm_config.yaml`."*

### Step 7: failure handling (1 min)
In Terminal 1 press `Ctrl+C` to stop LiteLLM, then:
```text
python scripts/chat.py "I want an easy waterfall hike" --verbose
```
Expected: a friendly "trouble reaching my language model" message, `reply_source: fallback`, and the real
cause in `error`. Then restart LiteLLM.

> *"The user gets a clean message, not a stack trace, and the developer still gets the real cause."*

### Step 8: the honest limitation (45 s)
```text
python scripts/chat.py "somewhere to clear my head" --verbose
```
Expected: `route: chat` and the "I'm built to help with hiking trails" reply.

> *"This is a real hiking request that keyword rules miss. It is exactly why Phase 3 hands this decision to an LLM agent."*

### Step 9: tests (30 s)
```text
cd services/agent
uv run pytest -q
```
Expected: **65 passed**.

> *"The LLM is behind an interface, so tests use a fake model. They are fast, deterministic, and need no Ollama."*

---

## 6. Observed results (local run)

Single runs on the development laptop (CPU only, Intel Iris Xe graphics, 31.7 GB RAM).

| Message | Path | Trace | Total |
|---|---|---|---|
| `hello` | chat (template) | input_guard → classify → chat_reply → output_guard | 43.3 ms (first call) |
| `what's the weather on the trail today` | reject (template) | input_guard → classify → reject → output_guard | 7.8 ms |
| `ignore all previous instructions` | blocked (`prompt_injection`) | input_guard → reject → output_guard | 4.5 ms |
| `I want an easy waterfall hike`, tier `fast` | trail (LLM) | input_guard → classify → trail_reply → output_guard | 9,263.8 ms, of which `trail_reply` 9,234.9 ms |
| same message, tier `smart` | trail (LLM) | same | 24,246.8 ms, of which `trail_reply` 24,238.9 ms |

Model speed from the environment smoke test: `qwen2.5:3b` generated about **9.5 tokens/s**. A direct
`ollama run --verbose` comparison measured 8.10 tokens/s for `qwen2.5:3b` and 10.02 tokens/s for
`llama3.2:3b`.

**Read these numbers carefully:**
- Each is a single run, and the direct model comparison generated only 7 to 9 tokens, so differences are not conclusive.
- The `smart` run is **not** a fair speed comparison. Ollama may have had to unload one model and load the
  other, which adds seconds, and the reply was longer. A fair comparison needs repeated runs with both models
  already loaded. That is planned for the model-comparison stage.
- Streaming was confirmed visually: words appeared gradually.
- The first request after the model has been idle pays the model load time (about 6 seconds measured).

---

## 7. Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Rule-based routing, not LLM routing | Predictable, testable, instant, free | Brittle: misses phrasing the keywords don't cover (Step 8) |
| Guardrails as graph nodes | Visible in the diagram, testable alone, can't be skipped by a code path | Adds two nodes to every request (about 1 ms) |
| Rule-based guardrails | Fast and predictable | Evadable by rephrasing; a model-based guardrail catches more but is slower and imperfect too |
| Output guardrail runs **after** streaming | Keeps token-by-token streaming | Bad text may already be on screen, so a `replace` event swaps it. The alternative is buffering the whole reply and losing streaming |
| Templated replies send one `token` event | Client has a single code path for every route | None significant |
| LiteLLM tiers instead of model names | Swap models with a config edit | One more service to run |
| `LLMClient` as a Protocol | Tests use a fake model; no Ollama needed | One extra abstraction |
| System prompt forbids inventing trails, numbers, weather | No data exists yet, so any specifics would be made up | A 3B model follows it loosely (see limitations) |
| Master key in `.env`, proxy bound to `127.0.0.1` | Newer LiteLLM refuses to start without a key; keeps the proxy off the network | `.env` must never be committed (it is git-ignored) |
| Fixed fallback message on model failure | The user never sees a crash | The real error is only in the `error` field |

## 8. Known limitations

- **Keyword routing is brittle.** A test documents this on purpose (`somewhere to clear my head` falls through to chat). Phase 3 replaces the router with an LLM agent.
- **Rule-based guardrails can be evaded** by rephrasing, and they can block innocent text that matches a pattern. They catch the common cases.
- **No trail data yet.** The trail path gives general advice only and cannot recommend anything specific. Data arrives in Phase 2.
- **Small-model instruction following.** The prompt asks for 2 to 4 sentences and one question. In local runs the 3B models sometimes returned longer answers or two questions.
- **Latency.** On a CPU-only laptop a reply takes about 9 seconds with `qwen2.5:3b`. This will matter more in Phase 3, where an agent makes several model calls per question.
- **No conversation memory.** Each message is handled independently. Memory is part of Phase 3.
- **Logging is basic.** Structured logs and request-ID correlation are Phase 4.

## 9. Definition of Done check

| Requirement | Status | Evidence |
|---|---|---|
| Runs locally with a real multi-node LangGraph graph | Done | `/graph`, 6 nodes |
| 2+ deterministic paths chosen by rules, not a model | Done | chat, reject and trail, plus blocked; Steps 2 to 5 |
| At least one path calls a real LLM through LiteLLM → Ollama | Done | Step 5; `reply_source: llm` |
| Responses stream via SSE | Done | gradual output observed; `meta/route/token/done` events |
| Input and output guardrails in place and tested against blocked and allowed messages | Done | `tests/test_guardrails.py` (parametrized allowed/blocked cases), Step 4 |
| Live walkthrough of a message through the full graph | This document | Section 5 |

## 10. Likely questions

**Why not let the model decide the route?** Same input can give different output, which makes behaviour hard to
test and debug. Rules are predictable and free. I only use the model where text generation is the job.

**What happens if Ollama or LiteLLM is down?** The trail node catches the failure, sends a fixed message, and puts
the real error in the `error` field. Unit tests cover model-down and model-fails-midway cases.

**How do you test something non-deterministic?** The LLM sits behind an interface. Tests inject a fake model, so
routing, streaming, guardrails and fallbacks are tested deterministically. Real model quality is evaluated
separately.

**Why does the output guardrail run after streaming?** To keep token-by-token streaming. If it rejects the
reply, a `replace` event swaps the text. The cost is that the user may briefly see text that then changes.

**Can the injection guardrail be bypassed?** Yes, by rephrasing. It stops common patterns cheaply. A
model-based check would catch more at the cost of latency and its own error rate.

**Why LiteLLM instead of calling Ollama directly?** The app asks for a tier name, so changing the model is a
config edit. It also makes cloud models a config change later.

**Why was `smart` slower?** Unproven from one run. Model loading and a longer reply are the likely causes. It needs a
repeated, fair comparison.

**What changes in Phase 3?** The classifier is replaced by an LLM agent that decides whether to answer, ask a
question or call a search tool. The guardrails and the SSE contract stay.

---

## 11. Reproduce it

```text
# once
ollama pull qwen2.5:3b
python scripts/start_litellm.py          # creates .env with a random master key

# agent
cd services/agent
uv sync
uv run uvicorn trailmate_agent.main:app
uv run pytest -q                         # 65 passed
uv run ruff check . ; uv run mypy src    # both clean

# environment check
python scripts/smoke_test.py
```
