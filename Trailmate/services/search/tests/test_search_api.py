"""Ingestion and search against a real PostgreSQL + pgvector, with a fake (non-semantic) embedder.

These prove the PIPELINE works. Whether results are *relevant* is measured with the real embedder by
eval/check_queries.py.
"""

import pytest
from conftest import SETTINGS, make_client
from fakes import DownEmbedder, FakeEmbedder
from sqlalchemy import func, select

from trailmate_search.ingest import ingest
from trailmate_search.models import Chunk, Trail

pytestmark = pytest.mark.usefixtures("test_db")


async def counts(session_factory):
    async with session_factory() as s:
        return (
            await s.scalar(select(func.count()).select_from(Trail)),
            await s.scalar(select(func.count()).select_from(Chunk)),
        )


async def test_ingest_loads_all_trails_and_chunks(ingested):
    assert await counts(ingested) == (16, 51)


async def test_ingest_is_idempotent(ingested):
    again = await ingest(ingested, FakeEmbedder(), SETTINGS.data_dir, SETTINGS.embedding_dim)
    assert (again.trails, again.chunks) == (16, 51)
    assert await counts(ingested) == (16, 51)  # replaced, not duplicated


async def test_ingest_rejects_wrong_dimension_and_leaves_database_untouched(session_factory):
    with pytest.raises(ValueError, match="384"):
        await ingest(
            session_factory, FakeEmbedder(dim=8), SETTINGS.data_dir, SETTINGS.embedding_dim
        )
    assert await counts(session_factory) == (0, 0)


async def test_ingest_leaves_database_untouched_when_embedder_fails(session_factory):
    with pytest.raises(Exception, match="refused"):
        await ingest(session_factory, DownEmbedder(), SETTINGS.data_dir, SETTINGS.embedding_dim)
    assert await counts(session_factory) == (0, 0)


async def test_health_counts(client):
    assert (await client.get("/health")).json() == {"status": "ok", "trails": 16, "chunks": 51}


async def test_search_returns_distinct_trails_best_first(client):
    # "waterfall" appears in several chunks of several trails; results must still be one row per trail
    r = await client.post("/search", json={"query": "waterfall", "top_k": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    slugs = [x["slug"] for x in results]
    assert len(results) == 5 and len(set(slugs)) == 5
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1 <= s <= 1 for s in scores)


async def test_search_finds_the_trail_whose_text_matches(client):
    r = await client.post("/search", json={"query": "plunge pool strong currents"})
    top = r.json()["results"][0]
    assert top["slug"] == "thunder-gorge-trail"
    assert "plunge pool" in top["excerpt"]
    assert top["matched_section"] == "Trail Notes"


async def test_search_result_carries_facts_and_evidence(client):
    top = (await client.post("/search", json={"query": "plunge pool strong currents"})).json()[
        "results"
    ][0]
    assert top["facts"] == {
        "difficulty": "Hard",
        "distance_km": 12.5,
        "elevation_gain_m": 890,
        "best_for": [
            "Full-day adventures",
            "waterfalls",
            "experienced hikers comfortable with stream crossings",
        ],
        "dogs_allowed": "Not recommended (stream crossings, exposed scrambling)",
    }


async def test_search_facts_match_the_trail_endpoint(client):
    hit = (await client.post("/search", json={"query": "sunset", "top_k": 1})).json()["results"][0]
    trail = (await client.get(f"/trails/{hit['slug']}")).json()
    assert hit["facts"] == trail["facts"]


async def test_top_k_limits_results_and_timing_is_reported(client):
    body = (await client.post("/search", json={"query": "easy walk", "top_k": 2})).json()
    assert len(body["results"]) == 2
    assert set(body["timing_ms"]) == {"embed", "search", "total"}


async def test_unknown_strategy_returns_no_results(client):
    r = await client.post("/search", json={"query": "easy walk", "strategy": "does-not-exist"})
    assert r.status_code == 200 and r.json()["results"] == []


@pytest.mark.parametrize(
    "payload", [{"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 11}, {}]
)
async def test_invalid_search_requests_are_rejected(client, payload):
    assert (await client.post("/search", json=payload)).status_code == 422


async def test_embedder_outage_gives_503_with_consistent_error_shape(ingested):
    async with make_client(ingested, DownEmbedder()) as c:
        r = await c.post("/search", json={"query": "easy walk"})
    assert r.status_code == 503
    assert r.json() == {
        "error": {"status": 503, "message": "The embedding service is unavailable."}
    }


async def test_trails_endpoints(client):
    assert len((await client.get("/trails")).json()) == 16
    one = (await client.get("/trails/cascade-falls-loop")).json()
    assert one["name"] == "Cascade Falls Loop" and one["facts"]["distance_km"] == 3.2
    assert (await client.get("/trails/no-such-trail")).status_code == 404
