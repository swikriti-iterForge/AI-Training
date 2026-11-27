"""The search tool: what the model is told about it, how its arguments are validated, and the client.

The model never touches the database. It can only ASK for `search_trails(query)`. Our code validates
the arguments, calls the search service over HTTP with a timeout, and passes the result on.
"""

from typing import Any, Protocol

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
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    terms_found: list[str] = []  # which of the requested terms this trail's text really mentions


class SearchResults(BaseModel):
    query: str
    results: list[TrailHit]


class TrailInfo(BaseModel):
    slug: str
    name: str
    tagline: str | None
    facts: Facts


class SearchUnavailable(Exception):
    """The search service is down, too slow, or returned something we cannot read."""


class TrailSearch(Protocol):
    async def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        mode: str | None = None,
        difficulty: list[str] | None = None,
        exclude: list[str] | None = None,
        terms: list[str] | None = None,
        min_km: float | None = None,
        max_km: float | None = None,
    ) -> SearchResults: ...

    async def get_trail(self, slug: str) -> TrailInfo: ...


class HttpTrailSearch:
    """Calls POST {base_url}/search on trailmate-search."""

    def __init__(self, base_url: str, timeout_s: float = 20.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        mode: str | None = None,
        difficulty: list[str] | None = None,
        exclude: list[str] | None = None,
        terms: list[str] | None = None,
        min_km: float | None = None,
        max_km: float | None = None,
    ) -> SearchResults:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        for key, value in (
            ("mode", mode),
            ("difficulty", difficulty),
            ("exclude", exclude),
            ("terms", terms),
            ("min_distance_km", min_km),
            ("max_distance_km", max_km),
        ):
            if (
                value is not None and value != []
            ):  # only send what is set, so an older search service still understands us
                payload[key] = value
        try:
            response = await self._client.post("/search", json=payload)
            response.raise_for_status()
            return SearchResults.model_validate(response.json())
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise SearchUnavailable(f"{type(exc).__name__}: {exc}"[:200]) from exc

    async def get_trail(self, slug: str) -> TrailInfo:
        try:
            response = await self._client.get(f"/trails/{slug}")
            response.raise_for_status()
            return TrailInfo.model_validate(response.json())
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise SearchUnavailable(f"{type(exc).__name__}: {exc}"[:200]) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
