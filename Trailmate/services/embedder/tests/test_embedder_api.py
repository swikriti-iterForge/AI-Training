"""Embedder API tests with a fake encoder (no torch, no model download)."""

import math

import pytest
from httpx import ASGITransport, AsyncClient

from trailmate_embedder.main import create_app


class FakeEncoder:
    name = "fake-encoder"
    dimensions = 8

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [float((len(text) + i) % 5 + 1) for i in range(self.dimensions)]
            norm = math.sqrt(sum(v * v for v in vec))
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
async def client():
    app = create_app(encoder=FakeEncoder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_embed_returns_one_vector_per_text(client):
    r = await client.post("/embed", json={"texts": ["easy waterfall hike", "hard scramble"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-encoder" and body["dimensions"] == 8
    assert len(body["embeddings"]) == 2
    assert all(len(v) == 8 for v in body["embeddings"])


async def test_vectors_are_normalised(client):
    vec = (await client.post("/embed", json={"texts": ["anything"]})).json()["embeddings"][0]
    assert math.isclose(sum(x * x for x in vec), 1.0, rel_tol=1e-6)


async def test_same_text_gives_same_vector(client):
    a = (await client.post("/embed", json={"texts": ["x"]})).json()["embeddings"][0]
    b = (await client.post("/embed", json={"texts": ["x"]})).json()["embeddings"][0]
    assert a == b


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},  # nothing to embed
        {"texts": [""]},  # empty text
        {"texts": ["x"] * 65},  # too many in one call
        {"texts": ["x" * 5001]},  # one text too long
        {},  # missing field
    ],
)
async def test_invalid_requests_are_rejected(client, payload):
    assert (await client.post("/embed", json=payload)).status_code == 422


async def test_health_reports_model(client):
    body = (await client.get("/health")).json()
    assert body == {"status": "ok", "model": "fake-encoder", "dimensions": 8}
