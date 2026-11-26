"""The only place that talks to a model - always through the LiteLLM proxy.

The app asks for a *tier* ("fast", "smart"). Which real model sits behind a tier is decided
in infra/litellm_config.yaml, so swapping models never touches this code.

`LLMClient` is a Protocol (like a Java interface): tests plug in a fake, no model needed.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, cast

from openai import AsyncOpenAI

from trailmate_agent.config import Settings


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str  # the raw JSON text exactly as the model produced it; our code validates it


@dataclass(frozen=True)
class Completion:
    content: str
    tool_calls: list[ToolCall]


class LLMClient(Protocol):
    def stream(self, messages: list[dict[str, str]], tier: str) -> AsyncIterator[str]:
        """Yield the reply piece by piece (token deltas)."""
        ...

    async def complete(
        self, messages: list[dict[str, str]], tier: str, tools: list[dict[str, Any]]
    ) -> Completion:
        """One non-streaming call where the model may answer in text OR ask to call a tool."""
        ...


class LiteLLMClient:
    def __init__(self, settings: Settings) -> None:
        # LiteLLM exposes the OpenAI API, so the OpenAI SDK works unchanged against it.
        self._client = AsyncOpenAI(
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # fail fast: the graph has its own fallback reply
        )

    async def stream(self, messages: list[dict[str, str]], tier: str) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=tier,
            messages=cast(Any, messages),  # plain {"role","content"} dicts are what the API wants
            stream=True,
            temperature=0.3,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete(
        self, messages: list[dict[str, str]], tier: str, tools: list[dict[str, Any]]
    ) -> Completion:
        response = await self._client.chat.completions.create(
            model=tier,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
            tool_choice="auto",
            temperature=0,  # decisions should be repeatable
            max_tokens=200,
        )
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            args = function.arguments
            calls.append(
                ToolCall(function.name, args if isinstance(args, str) else json.dumps(args))
            )
        return Completion(content=message.content or "", tool_calls=calls)
