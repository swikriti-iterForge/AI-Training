"""Changing your mind mid-chat: "okay show some easy ones" after an alpine search.

A real conversation got "I couldn't find a trail that matches that closely" twice, because the remembered
features ("alpine", "snow") were added to the new search, and no Easy trail is alpine. These tests reproduce it."""

import pytest
from agent_fakes import Conversation, FakeSearch, ScriptedLLM, event, hit
from test_graph_api import done, names, streamed_text

from trailmate_agent.answers import NO_MATCH
from trailmate_agent.config import Settings
from trailmate_agent.llm import Completion, ToolCall
from trailmate_agent.prefs import is_filter_only


def alpine():
    return hit(
        "timberline-pass", "Timberline Pass", 0.53, "Hard", 14.0, 1100, terms=("alpine", "snow")
    )


def easy(score):
    return [
        hit("silver-lake-circuit", "Silver Lake Circuit", score, "Easy", 4.0, 40),
        hit("hidden-spring-trail", "Hidden Spring Trail", score - 0.05, "Easy", 3.0, 60),
    ]


def like_the_real_data(query, difficulty, exclude):
    """Easy trails match a plain 'easy trails' query well, but score badly if the query is also about alpine snow."""
    if difficulty == ["Easy"]:
        return easy(0.2 if "alpine" in query else 0.5)
    return [alpine()]


def asks_for(query):
    return ScriptedLLM(
        decide=Completion("", [ToolCall("search_trails", f'{{"query": "{query}"}}')])
    )


async def test_easy_ones_after_an_alpine_search_leaves_the_alpine_part_out():
    c = Conversation(llm=asks_for("easy trails"), search=FakeSearch(by_query=like_the_real_data))
    first = await c.say("an alpine trail with snow")
    assert event(first, "results")[0]["trails"][0]["name"] == "Timberline Pass"

    events = await c.say("okay show some easy ones")
    text = streamed_text(events)
    assert done(events)["decision"] == "recommend" and "couldn't find" not in text
    assert [t["name"] for t in event(events, "results")[0]["trails"]] == [
        "Silver Lake Circuit",
        "Hidden Spring Trail",
    ]
    assert "I left alpine and snow out of this search" in text  # honest about what changed
    assert (
        "Why it fits: easy: it is rated Easy." in text and "Not matched: alpine" not in text
    )  # no clutter about it
    queries = [call["query"] for call in c.search.calls[-2:]]
    assert (
        "alpine" in queries[0] and "alpine" not in queries[1]
    )  # first tried with it, then without
    assert done(events)["tool_calls"][0]["query"] == "easy trails"


async def test_the_conversation_moves_on_so_alpine_is_not_carried_forever():
    c = Conversation(llm=asks_for("easy trails"), search=FakeSearch(by_query=like_the_real_data))
    await c.say("an alpine trail with snow")
    await c.say("okay show some easy ones")
    await c.say("why? show me easy ones")  # the second screenshot message
    assert c.search.calls[-1]["terms"] in ([], None)  # alpine and snow are gone from the session
    assert "alpine" not in c.search.calls[-1]["query"]


async def test_a_request_that_is_only_a_filter_is_never_refused_for_a_low_score():
    weak_everywhere = FakeSearch(
        by_query=lambda q, d, e: easy(0.1)
    )  # a content-free query matches nothing well
    c = Conversation(llm=asks_for("easy ones"), search=weak_everywhere)
    events = await c.say("okay show some easy ones")
    assert done(events)["decision"] == "recommend" and streamed_text(events) != NO_MATCH
    assert "results" in names(events)


async def test_a_request_with_real_content_and_a_weak_match_is_still_refused():
    weak_everywhere = FakeSearch(by_query=lambda q, d, e: easy(0.1))
    c = Conversation(llm=asks_for("easy trails with a glacier"), search=weak_everywhere)
    events = await c.say("easy trails with a glacier")
    assert (
        streamed_text(events) == NO_MATCH
        and done(events)["decision"] == "no_match"
        and "results" not in names(events)
    )


async def test_if_leaving_the_features_out_does_not_help_it_still_refuses_honestly():
    c = Conversation(
        llm=asks_for("easy trails"),
        search=FakeSearch(by_query=lambda q, d, e: easy(0.1) if d else [alpine()]),
    )
    await c.say("an alpine trail with snow")
    events = await c.say("easy trails with a glacier please")  # content, weak either way
    assert streamed_text(events) == NO_MATCH


async def test_good_matches_keep_the_remembered_features():
    """Dropping is a fallback for weak results. A request that works must keep what was said earlier."""
    c = Conversation(
        search=FakeSearch(
            by_query=lambda q, d, e: [hit("lake", "Lake Loop", 0.6, "Easy", 3, 30, terms=("lake",))]
        )
    )
    await c.say("an easy lake walk")
    events = await c.say("something shorter")
    assert c.search.calls[-1]["terms"] == ["lake"] and "left" not in streamed_text(events)


async def test_the_cutoff_setting_still_applies_to_ordinary_requests():
    c = Conversation(
        search=FakeSearch(by_query=lambda q, d, e: easy(0.3)), settings=Settings(min_relevance=0.5)
    )
    assert done(await c.say("an easy waterfall hike"))["decision"] == "no_match"


@pytest.mark.parametrize(
    "message, difficulty, expected",
    [
        ("okay show some easy ones", "Easy", True),
        ("why? show me easy ones", "Easy", True),
        ("not too hard please", "Easy", True),
        ("yes, harder", "Hard", True),
        ("easy waterfall hike", "Easy", False),  # says WHAT to see
        ("easy trails with a lake", "Easy", False),
        ("show me more", "Easy", False),
        (
            "okay show some easy ones",
            None,
            False,
        ),  # no difficulty was understood: nothing to filter by
    ],
)
def test_is_filter_only(message, difficulty, expected):
    assert is_filter_only(message, difficulty) is expected


# ---- an earlier feature that no result mentions ----
async def test_an_earlier_feature_no_result_mentions_is_left_out_even_when_the_scores_are_fine():
    """Seen over real HTTP: the easy trails scored well, yet every one was listed as 'Not matched: alpine, snow'."""
    c = Conversation(
        llm=asks_for("easy trails"),
        search=FakeSearch(by_query=lambda q, d, e: easy(0.5) if d else [alpine()]),
    )
    await c.say("an alpine trail with snow")
    events = await c.say("okay show some easy ones")
    text = streamed_text(events)
    assert "I left alpine and snow out of this search" in text and "Not matched: alpine" not in text
    assert (
        "Using what you said earlier" not in text and done(events)["memory_note"] is None
    )  # no contradicting note
    await c.say("something longer")
    assert c.search.calls[-1]["terms"] in ([], None)  # and the conversation has moved on


async def test_a_feature_typed_in_this_message_is_never_left_out():
    c = Conversation(
        llm=asks_for("easy walk"), search=FakeSearch(by_query=lambda q, d, e: easy(0.5))
    )
    events = await c.say(
        "an easy lake walk"
    )  # no result mentions a lake, but the user asked for one just now
    text = streamed_text(events)
    assert "left" not in text and "Not matched: lake (its description doesn't mention it)" in text
    assert c.search.calls[-1]["terms"] == ["lake"]


async def test_show_me_more_never_changes_what_the_search_is_about():
    first = [alpine()]
    c = Conversation(
        llm=asks_for("alpine trails"),
        search=FakeSearch(by_query=lambda q, d, e: easy(0.5) if e else first),
    )
    await c.say("an alpine trail with snow")
    events = await c.say("show me more")
    assert "left" not in streamed_text(events) and c.search.calls[-1]["terms"] == ["alpine", "snow"]


async def test_the_note_still_names_what_was_used_when_only_some_features_are_dropped():
    c = Conversation(
        llm=asks_for("easy trails"),
        search=FakeSearch(by_query=lambda q, d, e: easy(0.5) if d else [alpine()]),
    )
    await c.say("I love easy trails")  # saved: easy trails
    await c.say("an alpine trail with snow", session_id="two")
    events = await c.say(
        "okay show some options", session_id="two"
    )  # no difficulty said: the saved one applies
    text = streamed_text(events)
    assert (
        "I left alpine and snow out" in text
        and "(Using what I remember about you: easy trails.)" in text
    )
    assert "alpine, snow" not in text.replace("I left alpine and snow out", "")
