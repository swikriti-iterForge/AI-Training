"""FastAPI app: POST /chat streams the answer as Server-Sent Events.

SSE events (every path emits the same sequence, so the client has one code path):
  meta     {request_id, tier}                      first
  route    {path}                                  which branch the message took
  token    {text}                                  a piece of the reply (templated replies: ONE token)
  replace  {text}                                  a guardrail swapped the text already streamed
  done     {category, reply_source, trace, timings_ms, total_ms, ...}
  error    {message}                               only on an unexpected crash
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from trailmate_agent.config import Settings, get_settings
from trailmate_agent.graph import build_graph
from trailmate_agent.guardrails import load_config
from trailmate_agent.llm import LiteLLMClient, LLMClient

logger = logging.getLogger("trailmate.agent")


class ChatRequest(BaseModel):
    # The guardrail (not the schema) enforces the 1000-char limit, so users get a friendly
    # message; this larger cap just protects the server from huge payloads.
    message: str = Field(max_length=10_000)
    model: str | None = Field(default=None, description="Model tier, e.g. 'fast' or 'smart'")


def sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data)}


def create_app(llm: LLMClient | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    llm = llm or LiteLLMClient(settings)
    graph = build_graph(llm, load_config(settings.guardrail_config_path))
    app = FastAPI(title="TrailMate agent", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "tiers": settings.tiers, "default_tier": settings.default_tier}

    @app.get("/graph", response_class=PlainTextResponse)
    async def graph_diagram() -> str:
        """The workflow as a Mermaid diagram - paste it into https://mermaid.live to see it."""
        return graph.get_graph().draw_mermaid()

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
        tier = body.model or settings.default_tier
        if tier not in settings.tiers:
            raise HTTPException(
                422, f"Unknown model tier '{tier}'. Choose one of {settings.tiers}."
            )
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]

        async def events() -> AsyncIterator[dict[str, str]]:
            started = time.perf_counter()
            yield sse("meta", {"request_id": request_id, "tier": tier})
            final: dict[str, Any] = {}
            routed = False
            try:
                async for mode, chunk in graph.astream(
                    {"message": body.message, "tier": tier}, stream_mode=["custom", "values"]
                ):
                    if mode == "custom":  # token / replace, written by a node
                        yield sse(chunk["type"], {"text": chunk["text"]})
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
                "request %s %s via %s in %sms", request_id, final.get("category"), tier, total_ms
            )
            yield sse(
                "done",
                {
                    "request_id": request_id,
                    "tier": tier,
                    "category": final.get("category"),
                    "blocked": bool(final.get("blocked")),
                    "block_reason": final.get("block_reason") or None,
                    "reply_source": final.get("reply_source"),
                    "error": final.get("error"),
                    "trace": final.get("trace", []),
                    "timings_ms": final.get("timings", {}),
                    "total_ms": total_ms,
                },
            )

        return EventSourceResponse(events(), headers={"X-Request-ID": request_id})

    return app


app = create_app()  # run with: uvicorn trailmate_agent.main:app
