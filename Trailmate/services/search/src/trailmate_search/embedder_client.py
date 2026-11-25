"""Talks to the embedder service. Behind a Protocol (Java: an interface), so tests plug in a fake."""

from typing import Protocol

import httpx


class EmbedderUnavailable(Exception):
    """The embedder is down, too slow, or returned something unusable."""


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HttpEmbedder:
    """Calls POST {base_url}/embed. Spring comparison: a Feign client, written by hand."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post("/embed", json={"texts": texts})
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbedderUnavailable(f"embedder call failed: {exc}") from exc
        if len(embeddings) != len(texts):
            raise EmbedderUnavailable("embedder returned the wrong number of vectors")
        return embeddings

    async def aclose(self) -> None:
        await self._client.aclose()
