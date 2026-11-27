"""'short' and 'long': how long a trail is.

Two real messages were refused: "hi, i want short and long" and "short and challenging". They are about HOW LONG
and HOW HARD, with nothing about what to see, so the meaning-based search scored them low. Every trail has a
distance, so these are filters now. The limits come from the data: every Easy trail is <= 4 km, every Hard >= 8.2 km."""

import pytest
from agent_fakes import Conversation, FakeSearch, ScriptedLLM, hit
from test_graph_api import done, names, streamed_text

from trailmate_agent.answers import CLARIFY_QUESTION, explain_reply
from trailmate_agent.config import Settings
from trailmate_agent.llm import Completion, ToolCall
from trailmate_agent.prefs import Prefs, extract_prefs, merge_prefs, only_preference_words
from trailmate_agent.store import FilePreferenceStore

SHORT_EASY = hit("willow-creek-wander", "Willow Creek Wander", 0.3, "Easy", 2.1, 15)
LONG_HARD = hit("timberline-pass", "Timberline Pass", 0.3, "Hard", 14.0, 1100)
MID = hit("fern-valley-loop", "Fern Valley Loop", 0.3, "Moderate", 6.0, 200)
THE_DATA = [SHORT_EASY, MID, LONG_HARD]


def search_like_the_data(call):
    """A fake search that really applies the difficulty and distance limits it is given."""
    out = THE_DATA
    if call["difficulty"]:
        out = [h for h in out if h.facts.difficulty in call["difficulty"]]
    if call["max_km"] is not None:
        out = [h for h in out if h.facts.distance_km <= call["max_km"]]
    if call["min_km"] is not None:
        out = [h for h in out if h.facts.distance_km >= call["min_km"]]
    return out


def asks_for(query):
    return ScriptedLLM(
        decide=Completion("", [ToolCall("search_trails", f'{{"query": "{query}"}}')])
    )


# ---------- reading the words ----------
@pytest.mark.parametrize(
    "message, length",
    [
        ("a short walk", "short"),
        ("something quick", "short"),
        ("a long hike", "long"),
        ("a full-day outing", "long"),
        ("not too long please", "short"),
        ("not too short", "long"),
        ("short and long", None),  # both: no preference, like 'easy or hard'
        ("an easy waterfall hike", None),
    ],
)
def test_length_words(message, length):
    assert extract_prefs(message).length == length


def test_length_is_merged_like_difficulty():
    earlier = Prefs(length="short", terms=("lake",))
    assert merge_prefs(earlier, extract_prefs("something longer")).length == "long"
    assert merge_prefs(earlier, extract_prefs("another lake")).length == "short"


@pytest.mark.parametrize(
    "message, only",
    [
        ("hi, i want short and long", True),
        ("short and challenging", True),
        ("okay show some easy ones", True),
        ("a short waterfall walk", False),
        ("an easy lake walk", False),
    ],
)
def test_only_preference_words(message, only):
    assert only_preference_words(message) is only


# ---------- the two screenshot messages ----------
async def test_short_and_long_asks_a_question_instead_of_refusing():
    llm, search = ScriptedLLM(), FakeSearch(by_call=search_like_the_data)
    c = Conversation(llm=llm, search=search)
    events = await c.say("hi, i want short and long")
    assert streamed_text(events) == CLARIFY_QUESTION and done(events)["decision"] == "clarify"
    assert (
        llm.complete_calls == [] and search.calls == []
    )  # no model, no search: there is nothing to search for


async def test_short_and_challenging_explains_that_no_such_trail_exists():
    c = Conversation(
        llm=asks_for("short challenging"), search=FakeSearch(by_call=search_like_the_data)
    )
    events = await c.say("short and challenging")
    text = streamed_text(events)
    assert done(events)["decision"] == "recommend" and "I couldn't find" not in text
    assert text.startswith(
        "Note: I found no short trails with that, so these are the closest matches of any length."
    )
    assert "Best match: Timberline Pass (Hard, 14 km, 1100 m climb)." in text
    assert "Not matched: short (it is 14 km)." in text  # honest about exactly how it misses
    assert [(c_["difficulty"], c_["max_km"]) for c_ in c.search.calls] == [
        (["Hard"], 4.0),
        (["Hard"], None),
    ]  # tried, then loosened
    assert done(events)["tool_calls"][0]["relaxed_length"] is True


# ---------- the filter itself ----------
async def test_short_and_long_limits_reach_the_search_with_the_data_derived_numbers():
    c = Conversation(llm=asks_for("walk"), search=FakeSearch(by_call=search_like_the_data))
    await c.say("a short walk")
    assert (c.search.calls[-1]["min_km"], c.search.calls[-1]["max_km"]) == (None, 4.0)
    await c.say("a long hike", session_id="two")
    assert (c.search.calls[-1]["min_km"], c.search.calls[-1]["max_km"]) == (8.0, None)


async def test_the_limits_are_settings():
    c = Conversation(
        llm=asks_for("walk"),
        search=FakeSearch(by_call=search_like_the_data),
        settings=Settings(short_max_km=3.0, long_min_km=10.0),
    )
    await c.say("a short walk")
    assert c.search.calls[-1]["max_km"] == 3.0


async def test_a_short_easy_walk_just_works_and_is_explained():
    c = Conversation(llm=asks_for("easy walk"), search=FakeSearch(by_call=search_like_the_data))
    events = await c.say("a short easy walk")
    text = streamed_text(events)
    assert text.startswith("Best match: Willow Creek Wander (Easy, 2.1 km, 15 m climb).")
    assert (
        "Why it fits: easy: it is rated Easy; short: it is 2.1 km." in text and "Note:" not in text
    )
    assert "results" in names(events) and len(c.search.calls) == 1  # no loosening needed


@pytest.mark.parametrize("message", ["short and easy", "something short", "a long one"])
async def test_a_request_that_is_only_length_and_difficulty_is_never_refused_for_a_low_score(
    message,
):
    weak = FakeSearch(by_call=lambda call: [hit("a", "A", 0.05, "Easy", 2.0, 10)])
    c = Conversation(llm=asks_for("trails"), search=weak)
    assert done(await c.say(message))["decision"] == "recommend"


async def test_a_remembered_length_applies_to_later_requests_and_is_said():
    c = Conversation(llm=asks_for("trek"), search=FakeSearch(by_call=search_like_the_data))
    await c.say("I always like short walks")  # a standing statement: saved at once
    events = await c.say("Suggest a good trek", session_id="new-chat")
    assert c.search.calls[-1]["max_km"] == 4.0
    assert "(Using what I remember about you: short trails.)" in streamed_text(events)


# ---------- explanations and memory ----------
def test_explanations_say_how_far_a_trail_is_from_what_was_asked():
    text = explain_reply([SHORT_EASY, LONG_HARD], Prefs(length="short"))
    assert (
        "Why it fits: short: it is 2.1 km." in text
        and "Timberline Pass (Hard, 14 km, 1100 m climb; not matched: short)" in text
    )
    long = explain_reply([LONG_HARD], Prefs(length="long"))
    assert "long: it is 14 km" in long


def test_length_is_learned_like_difficulty(tmp_path):
    store = FilePreferenceStore(tmp_path / "p.json", min_mentions=2)
    store.learn("a", Prefs(length="short"))
    assert store.load("a").length is None  # once: only counted
    store.learn("a", Prefs(length="short"))
    assert store.load("a").length == "short" and store.summary("a")["learned"]["length"] == {
        "short": 2
    }
    for _ in range(2):
        store.learn("a", Prefs(length="long"))  # a tie is no favourite
    assert store.load("a").length is None


# ---------- when filters conflict, give up the OLD one, not the new one ----------
async def test_a_new_length_wins_over_a_difficulty_carried_from_earlier():
    """Seen over real HTTP: after 'short and challenging', 'something short' kept Hard and dropped the new word."""
    c = Conversation(llm=asks_for("trails"), search=FakeSearch(by_call=search_like_the_data))
    await c.say("short and challenging")
    events = await c.say("something short")
    text = streamed_text(events)
    assert done(events)["tool_calls"][0]["relaxed"] is True
    assert (
        "Best match: Willow Creek Wander (Easy, 2.1 km, 15 m climb)." in text
    )  # short, as asked just now
    assert text.startswith(
        "Note: I found no hard trails for this, so these are the closest matches of any difficulty."
    )
    assert (
        "Using what you said earlier" not in text and "Not matched: hard" not in text
    )  # no contradiction or clutter
    assert (c.search.calls[-2]["difficulty"], c.search.calls[-1]["difficulty"]) == (["Hard"], None)
    assert c.search.calls[-1]["max_km"] == 4.0  # the length the user just asked for was kept


async def test_the_dropped_difficulty_is_gone_from_the_conversation_afterwards():
    c = Conversation(llm=asks_for("trails"), search=FakeSearch(by_call=search_like_the_data))
    await c.say("short and challenging")
    await c.say("something short")
    await c.say("what else is there", session_id="s1")
    assert c.search.calls[-1]["difficulty"] is None


async def test_a_difficulty_typed_now_is_never_the_one_given_up():
    """'short and challenging' typed together: there is no such trail; the difficulty (a category) is kept."""
    c = Conversation(llm=asks_for("trails"), search=FakeSearch(by_call=search_like_the_data))
    events = await c.say("short and challenging")
    assert (
        done(events)["tool_calls"][0]["relaxed_length"] is True
        and done(events)["tool_calls"][0]["relaxed"] is False
    )


# ---------- conversational corrections must not negate the new preference ----------
@pytest.mark.parametrize(
    "message, expected",
    [
        ("no i want easy peaceful ones", "Easy"),
        ("no, I want easy peaceful one", "Easy"),
        ("actually easy ones", "Easy"),
        ("not too hard", "Easy"),
        ("not easy", "Hard"),
        ("no hard trails", "Easy"),
    ],
)
def test_difficulty_negation_does_not_misread_corrections(message, expected):
    assert extract_prefs(message).difficulty == expected
