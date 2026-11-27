"""FastAPI app: POST /chat streams the answer as Server-Sent Events.

Two modes run the same message through different graphs, so you can compare them:
  workflow  Phase 1: rule-based routing, the model only writes generic text (no trail data)
  agent     Phase 3: the model decides when to call the search tool; answers use real trail data

SSE events (every path emits the same core sequence, so the client has one code path):
  meta     {request_id, tier, mode, session_id}    first. Send session_id back to continue the conversation
                                                   (with a user_id, the conversation is also saved: see /conversations)
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
from pathlib import Path as FilePath
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import FileResponse, PlainTextResponse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from trailmate_agent.agent_graph import build_agent_graph, fresh_turn
from trailmate_agent.answers import asked_for
from trailmate_agent.config import Settings, get_settings
from trailmate_agent.conversations import ConversationStore, FileConversationStore
from trailmate_agent.graph import build_graph
from trailmate_agent.guardrails import load_config
from trailmate_agent.ids import ID_PATTERN
from trailmate_agent.llm import LiteLLMClient, LLMClient
from trailmate_agent.prefs import Prefs, describe
from trailmate_agent.store import FilePreferenceStore, PreferenceStore
from trailmate_agent.tools import HttpTrailSearch, TrailSearch

logger = logging.getLogger("trailmate.agent")

STATIC_DIR = FilePath(__file__).parent / "static"
Mode = Literal["workflow", "agent"]


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
    conversations: ConversationStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    llm = llm or LiteLLMClient(settings)
    search = search or HttpTrailSearch(settings.search_url, settings.search_timeout_s)
    store = store or FilePreferenceStore(settings.preference_file, settings.min_mentions)
    conversations = conversations or FileConversationStore(settings.conversation_dir)
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

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        """The chat UI: one HTML file that talks to POST /chat on this same server (so no CORS needed)."""
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "tiers": settings.tiers,
            "default_tier": settings.default_tier,
            "modes": list(graphs),
        }

    UserId = Annotated[str, Path(pattern=ID_PATTERN)]
    SessionId = Annotated[str, Path(pattern=ID_PATTERN)]

    @app.get("/memory/{user_id}")
    async def show_memory(user_id: UserId) -> dict[str, Any]:
        """What the agent knows about this user: `known` (applied by default), `explicit` (they said so) and
        `learned` (how often they asked for each thing)."""
        summary = store.summary(user_id)
        return {
            **summary,
            "known_text": describe(
                Prefs.from_dict(summary["known"])
            ),  # ready to show: "easy trails, waterfall"
            "asked_text": asked_for(
                summary["learned"]
            ),  # "waterfall (4 times), easy trails (3 times)"
        }

    @app.delete("/memory/{user_id}")
    async def erase_memory(user_id: UserId) -> dict[str, bool]:
        return {"deleted": store.delete(user_id)}

    @app.get("/conversations/{user_id}")
    async def list_conversations(user_id: UserId) -> list[dict[str, Any]]:
        """This user's saved chats, newest first."""
        return conversations.list_chats(user_id)

    @app.get("/conversations/{user_id}/{session_id}")
    async def get_conversation(user_id: UserId, session_id: SessionId) -> dict[str, Any]:
        chat = conversations.get(user_id, session_id)
        if chat is None:
            raise HTTPException(404, "No such conversation")
        return chat

    @app.delete("/conversations/{user_id}/{session_id}")
    async def delete_conversation(user_id: UserId, session_id: SessionId) -> dict[str, bool]:
        return {"deleted": conversations.delete(user_id, session_id)}

    @app.delete("/conversations/{user_id}")
    async def delete_all_conversations(user_id: UserId) -> dict[str, int]:
        return {"deleted": conversations.delete_all(user_id)}

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
            # the thread id includes the user, so one user can never continue another user's session
            run_config: dict[str, Any] = {
                "configurable": {"thread_id": f"{body.user_id or '-'}:{session_id}"}
            }
        else:
            turn_input, run_config = {"message": body.message, "tier": tier}, {}

        async def events() -> AsyncIterator[dict[str, str]]:
            started = time.perf_counter()
            if body.mode == "agent" and body.user_id and body.session_id:
                # A reopened chat after a restart: its session memory (what was shown, preferences so far)
                # was lost with the RAM, so restore it from the saved conversation.
                if not (await graph.aget_state(run_config)).values:
                    saved = conversations.get(body.user_id, body.session_id)
                    if saved and saved.get("memory"):
                        await graph.aupdate_state(
                            run_config, {"memory": saved["memory"]}, as_node="output_guard"
                        )
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
            if body.mode == "agent" and body.user_id:
                shown = final.get("decision") in ("recommend", "disambiguate")
                try:
                    conversations.append_turn(
                        body.user_id,
                        session_id,
                        body.message,
                        final.get("reply", ""),
                        final.get("hits") if shown else None,
                        final.get("memory"),
                    )
                except OSError:
                    logger.exception(
                        "could not save conversation %s", session_id
                    )  # chatting must not break
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
