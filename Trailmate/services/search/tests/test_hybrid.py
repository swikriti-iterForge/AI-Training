"""Keyword search, hybrid fusion (RRF), filters, exclusions and term evidence. Real PostgreSQL + pgvector.

The fake embedder is word-based, so these prove the MECHANICS (ranking, fusion, filtering). Whether hybrid
search is actually BETTER on the real model is measured by eval/check_queries.py --compare.
"""

import pytest

from trailmate_search.retrieval import keyword_tokens, rrf

pytestmark = pytest.mark.usefixtures("test_db")


async def search(client, query, **extra):
    r = await client.post("/search", json={"query": query, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def slugs(body):
    return [r["slug"] for r in body["results"]]


# ---------- pure functions (no database) ----------
def test_rrf_rewards_trails_that_rank_well_in_both_lists():
    scores = rrf([[1, 2, 3], [3, 2, 9]])
    assert scores[2] > scores[1] and scores[2] > scores[9]  # 2 is 2nd in both lists
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)  # 3rd in one, 1st in the other
    assert set(scores) == {1, 2, 3, 9}


def test_rrf_with_a_single_list_keeps_its_order():
    scores = rrf([[7, 5, 8]])
    assert sorted(scores, key=lambda t: -scores[t]) == [7, 5, 8]


def test_keyword_tokens_are_safe_lowercase_words():
    assert keyword_tokens("Easy trail; DROP TABLE trails -- near water!") == [
        "easy",
        "trail",
        "drop",
        "table",
        "trails",
        "near",
        "water",
    ]
    assert keyword_tokens("   ") == []


# ---------- keyword mode ----------
async def test_keyword_mode_finds_exact_words_and_stems_them(client):
    body = await search(client, "plunge pool", mode="keyword", top_k=3)
    assert body["mode"] == "keyword" and slugs(body)[0] == "thunder-gorge-trail"
    stemmed = await search(
        client, "crowded", mode="keyword", top_k=10
    )  # 'crowded' must match text saying 'crowds'
    assert len(stemmed["results"]) >= 4


async def test_keyword_mode_returns_nothing_when_no_word_matches(client):
    assert (await search(client, "zzzz qqqq", mode="keyword"))["results"] == []


async def test_score_is_always_the_cosine_similarity_even_in_keyword_mode(client):
    body = await search(client, "plunge pool", mode="keyword")
    assert all(-1 <= r["score"] <= 1 for r in body["results"])
    assert body["results"][0]["keyword_rank"] == 1 and body["results"][0]["vector_rank"] is not None


# ---------- hybrid mode ----------
async def test_hybrid_combines_both_rankings_and_exposes_the_evidence(client):
    body = await search(client, "plunge pool strong currents", mode="hybrid", top_k=5)
    top = body["results"][0]
    assert top["slug"] == "thunder-gorge-trail" and top["fused_score"] is not None
    assert top["vector_rank"] is not None and top["keyword_rank"] is not None
    fused = [r["fused_score"] for r in body["results"]]
    assert fused == sorted(fused, reverse=True)


async def test_fused_score_is_the_sum_of_reciprocal_ranks_from_both_lists(client):
    body = await search(client, "easy waterfall hike by a creek", mode="hybrid", top_k=6)
    both = 0
    for r in body["results"]:
        expected = sum(1 / (60 + rank) for rank in (r["vector_rank"], r["keyword_rank"]) if rank)
        assert r["fused_score"] == pytest.approx(expected, abs=1e-5)
        both += r["keyword_rank"] is not None
    assert (
        both >= 2
    )  # the query really exercised the keyword list, so dropping it from the fusion would fail above
    assert [r["fused_score"] for r in body["results"]] == sorted(
        (r["fused_score"] for r in body["results"]), reverse=True
    )


async def test_hybrid_still_returns_a_trail_that_only_the_vector_search_found(client):
    body = await search(client, "zzzz qqqq", mode="hybrid", top_k=3)  # no keyword hits at all
    assert len(body["results"]) == 3 and all(r["keyword_rank"] is None for r in body["results"])


async def test_default_mode_is_vector_and_unchanged(client):
    body = await search(client, "waterfall")
    assert body["mode"] == "vector" and all(r["fused_score"] is None for r in body["results"])


async def test_unknown_mode_is_rejected(client):
    assert (await client.post("/search", json={"query": "x", "mode": "magic"})).status_code == 422


# ---------- filters ----------
@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
async def test_difficulty_filter_applies_in_every_mode(client, mode):
    body = await search(client, "waterfall trail hike", mode=mode, difficulty=["Hard"], top_k=10)
    assert body["results"] and all(r["facts"]["difficulty"] == "Hard" for r in body["results"])


async def test_difficulty_filter_can_take_several_values(client):
    body = await search(client, "easy walk", difficulty=["Easy", "Easy-Moderate"], top_k=10)
    assert {r["facts"]["difficulty"] for r in body["results"]} <= {"Easy", "Easy-Moderate"}
    assert len(body["results"]) == 7  # 6 Easy + 1 Easy-Moderate in the data


async def test_difficulty_filter_with_no_match_returns_an_empty_list_not_an_error(client):
    assert (await search(client, "walk", difficulty=["Extreme"]))["results"] == []


async def test_exclude_removes_trails_already_shown(client):
    first = await search(client, "waterfall", top_k=3)
    again = await search(client, "waterfall", top_k=3, exclude=slugs(first))
    assert not set(slugs(first)) & set(slugs(again)) and len(again["results"]) == 3


# ---------- term evidence ----------
async def test_terms_found_reports_what_each_trail_actually_mentions(client):
    body = await search(
        client, "waterfall", top_k=10, difficulty=["Hard"], terms=["waterfall", "scramble"]
    )
    by_slug = {r["slug"]: r["terms_found"] for r in body["results"]}
    assert (
        "waterfall" in by_slug["thunder-gorge-trail"]
        and "scramble" in by_slug["thunder-gorge-trail"]
    )
    assert (
        "waterfall" not in by_slug["blackrock-summit"]
    )  # the trail the model wrongly credited with a waterfall
    assert "scramble" in by_slug["blackrock-summit"]


async def test_terms_are_limited_and_sanitised(client):
    ok = await search(client, "walk", terms=["lake", "x; drop table trails"])
    assert all("x; drop table trails" not in r["terms_found"] for r in ok["results"])
    assert (
        await client.post("/search", json={"query": "x", "terms": ["a"] * 9})
    ).status_code == 422


# ---------- distance limits ----------
@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
async def test_distance_limits_apply_in_every_mode(client, mode):
    short = await search(client, "easy trail walk hike", mode=mode, max_distance_km=4.0, top_k=10)
    assert short["results"] and all(r["facts"]["distance_km"] <= 4.0 for r in short["results"])
    long = await search(
        client, "hard trail hike scramble", mode=mode, min_distance_km=8.0, top_k=10
    )
    assert long["results"] and all(r["facts"]["distance_km"] >= 8.0 for r in long["results"])


async def test_distance_limits_form_a_range_and_combine_with_difficulty(client):
    middle = await search(
        client, "trail hike walk", min_distance_km=5.0, max_distance_km=6.0, top_k=10
    )
    assert middle["results"] and all(
        5.0 <= r["facts"]["distance_km"] <= 6.0 for r in middle["results"]
    )
    assert (await search(client, "trail hike", difficulty=["Hard"], max_distance_km=4.0))[
        "results"
    ] == []  # no short Hard trail
    assert (
        len(
            (
                await search(
                    client, "trail hike", difficulty=["Hard"], min_distance_km=8.0, top_k=10
                )
            )["results"]
        )
        == 5
    )


async def test_a_negative_distance_is_rejected(client):
    assert (
        await client.post("/search", json={"query": "x", "max_distance_km": -1})
    ).status_code == 422
