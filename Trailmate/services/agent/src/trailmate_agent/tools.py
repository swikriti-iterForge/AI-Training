"""The search tool: what the model is told about it, how its arguments are validated, and the client.

The model never touches the database. It can only ASK for `search_trails(query)`. Our code validates
the arguments, calls the search service over HTTP with a timeout, and passes the result on.
"""

from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError


class SearchTrailsArgs(BaseModel):
    query: str = Field(min_length=1, max_length=300)


# The tool description in the OpenAI format. The model reads this to decide when to call it.
SEARCH_TRAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_trails",
        "description": (
            "Search the trail database by meaning. Returns the best matching trails with their facts "
            "(difficulty, distance, climb, who it suits, dogs policy)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user wants, in plain words, keeping their preferences",
                }
            },
            "required": ["query"],
        },
    },
}


class Facts(BaseModel):
    difficulty: str
    distance_km: float
    elevation_gain_m: int
    best_for: list[str]
    dogs_allowed: str


class TrailHit(BaseModel):
    slug: str
    name: str
    score: float
    matched_section: str
    excerpt: str
    facts: Facts


class SearchResults(BaseModel):
    query: str
    results: list[TrailHit]


class SearchUnavailable(Exception):
    """The search service is down, too slow, or returned something we cannot read."""


class TrailSearch(Protocol):
    async def search(self, query: str, top_k: int = 3) -> SearchResults: ...


class HttpTrailSearch:
    """Calls POST {base_url}/search on trailmate-search."""

    def __init__(self, base_url: str, timeout_s: float = 20.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def search(self, query: str, top_k: int = 3) -> SearchResults:
        try:
            response = await self._client.post("/search", json={"query": query, "top_k": top_k})
            response.raise_for_status()
            return SearchResults.model_validate(response.json())
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise SearchUnavailable(f"{type(exc).__name__}: {exc}"[:200]) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
