"""FastAPI app: POST /chat streams the answer as Server-Sent Events.

Two modes run the same message through different graphs, so you can compare them:
  workflow  Phase 1: rule-based routing, the model only writes generic text (no trail data)
  agent     Phase 3: the model decides when to call the search tool; answers use real trail data

SSE events (every path emits the same core sequence, so the client has one code path):
  meta     {request_id, tier, mode, session_id}    first. Send session_id back to continue the conversation
  route    {path, reason}                          which branch the message took
  tool     {name, status, label, ms?}              agent mode: tool activity ("Searching hiking trails...")
  results  {trails: [...]}                         agent mode: the retrieved trails, with facts from the database
  token    {text}                                  a piece of the reply (templated replies: ONE token)
  replace  {text}                                  a guardrail swapped the text already streamed
  done     {category, decision, reply_source, tool_calls, trace, timings_ms, total_ms, ...}
  error    {message}                               only on an unexpected crash
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import PlainTextResponse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from trailmate_agent.agent_graph import build_agent_graph, fresh_turn
from trailmate_agent.config import Settings, get_settings
from trailmate_agent.graph import build_graph
from trailmate_agent.guardrails import load_config
from trailmate_agent.llm import LiteLLMClient, LLMClient
from trailmate_agent.store import FilePreferenceStore, PreferenceStore
from trailmate_agent.tools import HttpTrailSearch, TrailSearch

logger = logging.getLogger("trailmate.agent")

Mode = Literal["workflow", "agent"]
ID_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"  # session and user ids: short, simple, safe to use as keys


class ChatRequest(BaseModel):
    # The guardrail (not the schema) enforces the 1000-char limit, so users get a friendly
    # message; this larger cap just protects the server from huge payloads.
    message: str = Field(max_length=10_000)
    model: str | None = Field(default=None, description="Model tier, e.g. 'fast' or 'smart'")
    mode: Mode = Field(default="workflow", description="'workflow' (Phase 1) or 'agent' (Phase 3)")
    session_id: str | None = Field(
        default=None,
        pattern=ID_PATTERN,
        description="continues a conversation (agent mode); omit to start one",
    )
    user_id: str | None = Field(
        default=None,
        pattern=ID_PATTERN,
        description="who is asking; needed for long-term preferences",
    )


def sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data)}


def create_app(
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    search: TrailSearch | None = None,
    store: PreferenceStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    llm = llm or LiteLLMClient(settings)
    search = search or HttpTrailSearch(settings.search_url, settings.search_timeout_s)
    store = store or FilePreferenceStore(settings.preference_file)
    guardrails = load_config(settings.guardrail_config_path)
    graphs = {
        "workflow": build_graph(llm, guardrails),
        # short-term memory: the checkpointer keeps each session's state between turns (in RAM: lost on restart)
        "agent": build_agent_graph(llm, search, guardrails, settings, store, InMemorySaver()),
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if isinstance(search, HttpTrailSearch):
            await search.aclose()

    app = FastAPI(title="TrailMate agent", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "tiers": settings.tiers,
            "default_tier": settings.default_tier,
            "modes": list(graphs),
        }

    @app.get("/memory/{user_id}")
    async def show_memory(user_id: Annotated[str, Path(pattern=ID_PATTERN)]) -> dict[str, Any]:
        """What the agent has saved for this user (only what they asked it to remember)."""
        return store.load(user_id).to_dict()

    @app.delete("/memory/{user_id}")
    async def erase_memory(user_id: Annotated[str, Path(pattern=ID_PATTERN)]) -> dict[str, bool]:
        return {"deleted": store.delete(user_id)}

    @app.get("/graph", response_class=PlainTextResponse)
    async def graph_diagram(mode: Mode = "workflow") -> str:
        """The graph as a Mermaid diagram - paste it into https://mermaid.live to see it."""
        return graphs[mode].get_graph().draw_mermaid()

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
        tier = body.model or settings.default_tier
        if tier not in settings.tiers:
            raise HTTPException(
                422, f"Unknown model tier '{tier}'. Choose one of {settings.tiers}."
            )
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        graph = graphs[body.mode]
        session_id = body.session_id or uuid.uuid4().hex[:12]
        if body.mode == "agent":
            turn_input: dict[str, Any] = fresh_turn(body.message, tier, body.user_id)
            run_config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        else:
            turn_input, run_config = {"message": body.message, "tier": tier}, {}

        async def events() -> AsyncIterator[dict[str, str]]:
            started = time.perf_counter()
            yield sse(
                "meta",
                {
                    "request_id": request_id,
                    "tier": tier,
                    "mode": body.mode,
                    "session_id": session_id,
                },
            )
            final: dict[str, Any] = {}
            routed = False
            try:
                async for stream_kind, chunk in graph.astream(
                    turn_input, run_config, stream_mode=["custom", "values"]
                ):
                    if (
                        stream_kind == "custom"
                    ):  # token / replace / tool / results, written by a node
                        payload = dict(chunk)
                        yield sse(payload.pop("type"), payload)
                        continue
                    final = chunk  # full state after each step; the last one is the result
                    if not routed and (final.get("blocked") or final.get("category")):
                        routed = True
                        path = "blocked" if final.get("blocked") else final["category"]
                        yield sse(
                            "route", {"path": path, "reason": final.get("block_reason") or None}
                        )
            except Exception:
                logger.exception("request %s failed", request_id)
                yield sse("error", {"message": "Something went wrong on our side."})
                return
            total_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.info(
                "request %s %s/%s via %s in %sms",
                request_id,
                body.mode,
                final.get("category"),
                tier,
                total_ms,
            )
            yield sse(
                "done",
                {
                    "request_id": request_id,
                    "tier": tier,
                    "mode": body.mode,
                    "session_id": session_id,
                    "category": final.get("category") or None,
                    "decision": final.get("decision") or None,
                    "memory_note": final.get("memory_note") or None,
                    "blocked": bool(final.get("blocked")),
                    "block_reason": final.get("block_reason") or None,
                    "reply_source": final.get("reply_source"),
                    "error": final.get("error") or None,
                    "tool_calls": final.get("tool_calls", []),
                    "trace": final.get("trace", []),
                    "timings_ms": final.get("timings", {}),
                    "total_ms": total_ms,
                },
            )

        return EventSourceResponse(events(), headers={"X-Request-ID": request_id})

    return app


app = create_app()  # run with: uvicorn trailmate_agent.main:app
