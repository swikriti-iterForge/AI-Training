"""The only place that talks to a model - always through the LiteLLM proxy.

The app asks for a *tier* ("fast", "smart"). Which real model sits behind a tier is decided
in infra/litellm_config.yaml, so swapping models never touches this code.

`LLMClient` is a Protocol (like a Java interface): tests plug in a fake, no model needed.
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol, cast

from openai import AsyncOpenAI

from trailmate_agent.config import Settings


class LLMClient(Protocol):
    def stream(self, messages: list[dict[str, str]], tier: str) -> AsyncIterator[str]:
        """Yield the reply piece by piece (token deltas)."""
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
