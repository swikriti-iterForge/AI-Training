"""Phase 3 additions: reading preferences from words, code-built explanations, the difficulty filter, the
relevance cutoff, place names, and short-term (session) and long-term (saved) memory.

Everything here is deterministic: a scripted model, a fake search service, and the real API and graph."""

import pytest
from agent_fakes import (
    BLACKROCK,
    CLEAR,
    EXTENSION,
    GORGE,
    LOOP,
    Conversation,
    FakeSearch,
    InMemoryStore,
    ScriptedLLM,
    agent,
    event,
    hit,
    tmp_conversations,
)
from httpx import ASGITransport, AsyncClient
from test_graph_api import done, names, streamed_text

from trailmate_agent.answers import (
    CLARIFY_QUESTION,
    FORGET_REPLY,
    NEED_USER,
    NO_MATCH,
    NO_MORE,
    REMEMBER_HELP,
    explain_reply,
)
from trailmate_agent.config import Settings
from trailmate_agent.llm import Completion
from trailmate_agent.main import create_app
from trailmate_agent.prefs import (
    Prefs,
    extract_prefs,
    follow_up,
    merge_prefs,
    place_names,
    strip_location,
)
from trailmate_agent.store import FilePreferenceStore

UNIVERSE = [EXTENSION, LOOP, BLACKROCK, GORGE]  # best score first, like the real service


def by_filter(difficulty, exclude):
    """A fake search that really honours the filters it is given."""
    pool = [h for h in UNIVERSE if h.slug not in (exclude or [])]
    if difficulty:
        pool = [h for h in pool if h.facts.difficulty in difficulty]
    return pool[:3]


def test_defaults_are_the_safe_ones():
    s = Settings()
    assert s.explain_with_llm is False and s.min_relevance == 0.25 and s.search_mode == "vector"


# ============ reading preferences from words (no model) ============
@pytest.mark.parametrize(
    "message, difficulty, terms",
    [
        ("I want an easy waterfall hike", "Easy", ("waterfall",)),
        ("not too hard please", "Easy", ()),
        ("something harder", "Hard", ()),
        ("it shouldn't be too easy", "Hard", ()),
        ("easy or hard, whatever", None, ()),  # no single preference
        ("a moderate walk by a lake", "Moderate", ("lake",)),
        ("a challenging scramble with views", "Hard", ("scramble", "view")),
        ("waterfalls and wildflowers", None, ("waterfall", "wildflower")),
        ("show me something nice", None, ()),
    ],
)
def test_extract_prefs(message, difficulty, terms):
    p = extract_prefs(message)
    assert (p.difficulty, p.terms) == (difficulty, terms)


def test_dog_and_quiet_flags_are_read():
    p = extract_prefs("a quiet dog friendly walk")
    assert p.dogs and p.quiet and p.difficulty is None


def test_a_new_topic_replaces_old_features_but_a_refinement_keeps_them():
    earlier = Prefs(difficulty="Easy", terms=("waterfall",), dogs=True)
    refined = merge_prefs(earlier, extract_prefs("something harder"))
    assert refined.difficulty == "Hard" and refined.terms == ("waterfall",) and refined.dogs
    new_topic = merge_prefs(earlier, extract_prefs("an alpine trail with snow"))
    assert new_topic.terms == ("alpine", "snow") and new_topic.difficulty == "Easy"


@pytest.mark.parametrize(
    "message, expected",
    [
        ("show me more", ("more", None)),
        ("more options please", ("more", None)),
        ("another one", ("more", None)),
        ("anything else?", ("more", None)),
        ("something else", ("more", None)),
        ("the second one", ("pick", 2)),
        ("tell me about the first", ("pick", 1)),
        ("what about #3", ("pick", 3)),
        ("option 2", ("pick", 2)),
        ("more challenging please", None),  # a refinement, not "more of the same"
        ("it is my first time hiking trail", None),  # 'first' here is not an ordinal pick
        ("I want a long hike with more views", None),
        ("remember that I prefer easy trails", ("remember", None)),
        ("please remember I love waterfalls", ("remember", None)),
        ("forget my preferences", ("forget", None)),
        ("delete what you remember", ("forget", None)),
        ("what do you remember about me", ("recall", None)),
        ("show my preferences", ("recall", None)),
        ("start over", ("reset", None)),
        ("never mind", ("reset", None)),
    ],
)
def test_follow_up_commands(message, expected):
    assert follow_up(message, has_shown=True) == expected


def test_more_and_pick_need_something_to_refer_to():
    assert follow_up("show me more", has_shown=False) is None
    assert follow_up("the second one", has_shown=False) is None
    assert follow_up("forget my preferences", has_shown=False) == (
        "forget",
        None,
    )  # these never need results


@pytest.mark.parametrize(
    "query, message, cleaned",
    [
        (
            "easy waterfall hike near Mumbai",
            "Any easy waterfall hikes near Mumbai?",
            "easy waterfall hike",
        ),
        ("hike around Pune", "trails around Pune", "hike"),
        (
            "easy hike mumbai",
            "easy hike near Mumbai",
            "easy hike",
        ),  # the model lower-cased the place
        (
            "easy waterfall hike",
            "an easy waterfall hike",
            "easy waterfall hike",
        ),  # nothing to remove
        (
            "trail in the mountains",
            "a trail in the mountains",
            "trail in the mountains",
        ),  # not a place name
    ],
)
def test_strip_location(query, message, cleaned):
    assert strip_location(query, message) == cleaned


def test_place_names():
    assert (
        place_names("hikes near Navi Mumbai?") == ["Navi", "Mumbai"]
        and place_names("an easy hike") == []
    )


# ============ long-term store ============
def test_file_store_round_trip_and_isolation(tmp_path):
    store = FilePreferenceStore(tmp_path / "memory" / "prefs.json")
    assert store.load("a").is_empty  # nothing yet, and the file does not exist
    store.save("a", Prefs(difficulty="Easy", terms=("lake",), dogs=True))
    store.save("b", Prefs(difficulty="Hard"))
    assert (
        store.load("a") == Prefs(difficulty="Easy", terms=("lake",), dogs=True)
        and store.load("b").difficulty == "Hard"
    )
    assert store.delete("a") is True and store.delete("a") is False
    assert store.load("a").is_empty and store.load("b").difficulty == "Hard"
    assert not list(
        (tmp_path / "memory").glob("*.tmp")
    )  # the atomic write left no temp file behind


def test_file_store_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text("{not json", encoding="utf-8")
    store = FilePreferenceStore(path)
    assert store.load("a").is_empty
    store.save("a", Prefs(difficulty="Easy"))
    assert store.load("a").difficulty == "Easy"


# ============ explanations built by code ============
def test_explanation_says_what_matches_and_what_does_not():
    text = explain_reply(CLEAR, Prefs(difficulty="Easy", terms=("waterfall",)))
    assert text.startswith(
        "Best match: Cascade Falls Extension (Easy-Moderate, 5.5 km, 180 m climb)."
    )
    assert "Why it fits: waterfall: its description mentions it." in text
    assert (
        "Not matched: easy (it is rated Easy-Moderate)." in text
    )  # honest: the top hit is not strictly Easy
    assert "Cascade Falls Loop (Easy, 3.2 km, 90 m climb)" in text
    assert "Thunder Gorge Trail (Hard, 12.5 km, 890 m climb; not matched: easy)" in text


def test_a_trail_is_never_credited_with_a_feature_its_text_lacks():
    """The real failure: the model said Blackrock Summit fit 'a waterfall'. Its text never mentions one."""
    text = explain_reply(
        [BLACKROCK, GORGE], Prefs(difficulty="Hard", terms=("scramble", "waterfall"))
    )
    why = next(line for line in text.splitlines() if line.startswith("Why it fits"))
    assert "hard" in why and "scramble" in why and "waterfall" not in why
    assert "Not matched: waterfall (its description doesn't mention it)." in text


def test_explanation_reports_dogs_and_admits_what_the_data_cannot_say():
    text = explain_reply([LOOP], Prefs(dogs=True, quiet=True))
    assert "Dogs: Yes, on leash." in text and "Not in my data: how busy a trail is" in text
    assert "closest match to what you described" in explain_reply([LOOP], Prefs())


async def test_the_default_recommendation_costs_no_second_model_call():
    llm = ScriptedLLM()
    _, events = await agent(llm, FakeSearch(), "I want an easy waterfall hike", llm_writer=False)
    d = done(events)
    assert d["trace"] == [
        "input_guard",
        "triage",
        "decide",
        "run_search",
        "write_answer",
        "output_guard",
    ]
    assert d["reply_source"] == "template" and d["decision"] == "recommend" and d["error"] is None
    assert (
        len(llm.complete_calls) == 1 and llm.stream_calls == []
    )  # one call to decide, none to write
    assert "Best match: Cascade Falls Extension" in streamed_text(events)


# ============ filters, cutoff, places ============
async def test_the_difficulty_and_features_in_the_request_reach_the_search():
    search = FakeSearch()
    await agent(ScriptedLLM(), search, "I want an easy waterfall hike", llm_writer=False)
    call = search.calls[0]
    assert (
        call["difficulty"] == ["Easy"]
        and call["terms"] == ["waterfall"]
        and call["mode"] == "vector"
    )


async def test_when_no_trail_has_that_difficulty_the_filter_is_dropped_and_the_user_is_told():
    search = FakeSearch(hits=lambda difficulty, exclude: [] if difficulty else CLEAR)
    _, events = await agent(ScriptedLLM(), search, "a hard waterfall hike", llm_writer=False)
    text = streamed_text(events)
    assert text.startswith(
        "Note: I found no hard trails for this, so these are the closest matches of any difficulty."
    )
    assert [c["difficulty"] for c in search.calls] == [["Hard"], None]
    assert (
        done(events)["tool_calls"][0]["relaxed"] is True and done(events)["decision"] == "recommend"
    )


async def test_a_weak_match_is_not_recommended():
    weak = [
        hit("ridgeline-overlook", "Ridgeline Overlook", 0.144, "Moderate", 6.5, 420),
        hit("silver-lake-circuit", "Silver Lake Circuit", 0.122, "Easy", 4.0, 40),
    ]
    llm = ScriptedLLM(decide=Completion("", []))
    llm.decide = Completion(
        "",
        [
            __import__("trailmate_agent.llm", fromlist=["ToolCall"]).ToolCall(
                "search_trails", '{"query": "clear my head"}'
            )
        ],
    )
    _, events = await agent(
        llm, FakeSearch(hits=weak), "somewhere to clear my head", llm_writer=False
    )
    assert streamed_text(events) == NO_MATCH
    d = done(events)
    assert d["decision"] == "no_match" and "results" not in names(
        events
    )  # no trail cards for a non-answer
    assert d["trace"] == ["input_guard", "triage", "decide", "run_search", "output_guard"]


async def test_the_cutoff_is_a_setting():
    _, events = await agent(
        ScriptedLLM(),
        FakeSearch(),
        "I want an easy waterfall hike",
        llm_writer=False,
        settings=Settings(min_relevance=0.9),
    )
    assert done(events)["decision"] == "no_match"


async def test_place_names_are_removed_from_the_search_and_the_user_is_told_why():
    llm = ScriptedLLM(
        decide=Completion(
            "",
            [
                __import__("trailmate_agent.llm", fromlist=["ToolCall"]).ToolCall(
                    "search_trails", '{"query": "easy waterfall hike near Mumbai"}'
                )
            ],
        )
    )
    search = FakeSearch()
    _, events = await agent(llm, search, "Any easy waterfall hikes near Mumbai?", llm_writer=False)
    assert (
        "Mumbai" not in search.calls[0]["query"]
        and search.calls[0]["query"] == "easy waterfall hike"
    )
    assert streamed_text(events).startswith("Note: my trail data doesn't include locations")


async def test_a_model_written_reply_that_echoes_the_place_is_replaced():
    llm = ScriptedLLM(write=("This easy hike is perfect for families near Mumbai.",))
    _, events = await agent(
        llm, FakeSearch(), "Any easy waterfall hikes near Mumbai?", llm_writer=True
    )
    assert "perfect for families" not in streamed_text(events)
    assert done(events)["error"] == "grounding:place:Mumbai"


# ============ short-term memory: one conversation ============
async def test_show_me_more_skips_what_was_already_shown():
    c = Conversation(search=FakeSearch(hits=by_filter))
    first = await c.say("I want a waterfall hike")
    assert [t["slug"] for t in event(first, "results")[0]["trails"]] == [
        "cascade-falls-extension",
        "cascade-falls-loop",
        "blackrock-summit",
    ]

    second = await c.say("show me more")
    d = done(second)
    assert d["category"] == "more" and d["trace"][:3] == ["input_guard", "triage", "prepare_more"]
    assert c.search.calls[1]["exclude"] == [
        "cascade-falls-extension",
        "cascade-falls-loop",
        "blackrock-summit",
    ]
    assert [t["slug"] for t in event(second, "results")[0]["trails"]] == ["thunder-gorge-trail"]
    assert len(c.llm.complete_calls) == 1  # the model was not asked again: "more" is a command

    third = await c.say("anything else?")  # everything has now been shown
    assert streamed_text(third) == NO_MORE and "results" not in names(third)


async def test_the_second_one_gives_that_trails_details_from_the_database():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want a waterfall hike")
    events = await c.say("tell me about the second one")
    d = done(events)
    assert d["decision"] == "details" and d["tool_calls"] == [
        {"name": "get_trail", "slug": "cascade-falls-loop", "ok": True}
    ]
    text = streamed_text(events)
    assert (
        text.startswith("Cascade Falls Loop: Cascade Falls Loop tagline.")
        and "Easy, 3.2 km, 90 m climb" in text
    )
    assert len(c.llm.complete_calls) == 1


async def test_picking_a_number_that_was_not_shown_is_handled():
    c = Conversation(search=FakeSearch(hits=lambda difficulty, exclude: [LOOP]))
    await c.say("I want an easy hike")
    assert streamed_text(await c.say("the third one")).startswith("I only showed 1 trail.")


async def test_things_with_no_results_to_refer_to_are_not_treated_as_commands():
    c = Conversation()
    events = await c.say("show me more")  # nothing has been shown yet
    assert done(events)["category"] != "more"


async def test_a_refinement_keeps_the_earlier_features_and_says_so():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want an easy waterfall hike")
    events = await c.say("something harder")
    call = c.search.calls[1]
    assert call["difficulty"] == ["Hard"] and call["terms"] == [
        "waterfall"
    ]  # new difficulty, old feature
    assert "(Using what you said earlier: waterfall.)" in streamed_text(events)
    assert done(events)["memory_note"] == "(Using what you said earlier: waterfall.)"


async def test_a_new_topic_does_not_inherit_the_old_features():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want an easy waterfall hike")
    await c.say("an alpine trail with snow")
    assert c.search.calls[1]["terms"] == ["alpine", "snow"] and c.search.calls[1]["difficulty"] == [
        "Easy"
    ]


async def test_a_vague_message_is_not_vague_when_the_chat_already_says_what_is_wanted():
    c = Conversation(search=FakeSearch(hits=by_filter))
    assert (
        streamed_text(await c.say("Suggest a good trek")) == CLARIFY_QUESTION
    )  # nothing known yet
    await c.say("easy waterfall hike")
    events = await c.say("Suggest a good trek")
    assert done(events)["decision"] != "clarify" and c.search.calls[-1]["difficulty"] == ["Easy"]


async def test_sessions_do_not_share_memory():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want an easy waterfall hike", session_id="alice-chat")
    events = await c.say("show me more", session_id="bob-chat")
    assert done(events)["category"] != "more"
    assert (
        c.search.calls[-1]["difficulty"] is None and not c.search.calls[-1]["exclude"]
    )  # nothing leaked


async def test_memory_survives_a_chat_message_in_between():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want a waterfall hike")
    assert done(await c.say("Thanks!"))["decision"] == "chat"
    assert done(await c.say("show me more"))["category"] == "more"


async def test_nothing_from_a_failed_turn_leaks_into_the_next():
    search = FakeSearch(fail=True)
    c = Conversation(search=search)
    broken = await c.say("I want an easy waterfall hike")
    assert "SearchUnavailable" in done(broken)["error"] and done(broken)["decision"] == "error"
    search.fail = False
    ok = done(await c.say("I want an easy waterfall hike"))
    assert ok["error"] is None and ok["decision"] == "recommend" and len(ok["tool_calls"]) == 1
    assert ok["trace"] == [
        "input_guard",
        "triage",
        "decide",
        "run_search",
        "write_answer",
        "output_guard",
    ]  # not accumulated


async def test_start_over_clears_the_conversation_but_not_saved_preferences():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("remember that I prefer easy trails")
    await c.say("I want a waterfall hike")
    await c.say("start over")
    assert done(await c.say("show me more"))["category"] != "more"
    assert c.store.load("u1").difficulty == "Easy"


# ============ long-term memory: only what the user asks us to keep ============
async def test_remember_saves_and_a_new_session_uses_it_transparently():
    c = Conversation(search=FakeSearch(hits=by_filter))
    said = await c.say("remember that I prefer easy trails with waterfalls")
    assert streamed_text(said).startswith("Got it. I'll remember: easy trails, waterfall.")
    assert c.store.load("u1") == Prefs(difficulty="Easy", terms=("waterfall",))

    events = await c.say(
        "Suggest a good trek", session_id="a-brand-new-session"
    )  # vague, but we know what they like
    assert done(events)["decision"] != "clarify"
    assert c.search.calls[-1]["difficulty"] == ["Easy"] and c.search.calls[-1]["terms"] == [
        "waterfall"
    ]
    assert "(Using what I remember about you: easy trails, waterfall.)" in streamed_text(events)


async def test_what_do_you_remember_and_forget():
    c = Conversation()
    assert "not remembering anything" in streamed_text(await c.say("what do you remember about me"))
    await c.say("remember that I prefer hard trails")
    assert "What I remember about you: hard trails." in streamed_text(
        await c.say("what do you remember about me")
    )
    assert streamed_text(await c.say("forget my preferences")) == FORGET_REPLY
    assert c.store.load("u1").is_empty
    assert (
        streamed_text(await c.say("Suggest a good trek", session_id="fresh")) == CLARIFY_QUESTION
    )  # back to asking


async def test_remember_without_a_user_id_saves_nothing():
    c = Conversation()
    assert (
        streamed_text(await c.say("remember that I prefer easy trails", user_id=None)) == NEED_USER
    )
    assert c.store.data == {}


async def test_remember_with_nothing_to_remember_explains_what_works():
    c = Conversation()
    assert (
        streamed_text(await c.say("remember that I like turtles")) == REMEMBER_HELP
        and c.store.data == {}
    )


async def test_saved_preferences_belong_to_one_user():
    c = Conversation()
    await c.say("remember that I prefer easy trails", user_id="alice")
    assert c.store.load("bob").is_empty
    assert streamed_text(
        await c.say("what do you remember about me", user_id="bob", session_id="b1")
    ).startswith("I'm not remembering")


# ============ API surface ============
async def test_memory_endpoints_show_and_erase():
    store = InMemoryStore()
    store.save("alice", Prefs(difficulty="Easy", terms=("lake",)))
    app = create_app(
        llm=ScriptedLLM(),
        settings=Settings(),
        search=FakeSearch(),
        store=store,
        conversations=tmp_conversations(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/memory/alice")).json()["known"] == {
            "difficulty": "Easy",
            "terms": ["lake"],
            "dogs": False,
            "quiet": False,
            "length": None,
        }
        assert (await client.delete("/memory/alice")).json() == {"deleted": True}
        assert (await client.delete("/memory/alice")).json() == {"deleted": False}
        assert (await client.get("/memory/alice")).json()["known"]["difficulty"] is None
        assert (await client.get(f"/memory/{'x' * 65}")).status_code == 422


@pytest.mark.parametrize(
    "field, value", [("session_id", "has spaces"), ("user_id", "a/b"), ("session_id", "x" * 65)]
)
async def test_ids_are_validated(field, value):
    app = create_app(
        llm=ScriptedLLM(), settings=Settings(), search=FakeSearch(), store=InMemoryStore()
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/chat", json={"message": "hi", "mode": "agent", field: value})
    assert r.status_code == 422


async def test_the_session_id_is_issued_and_returned_so_the_client_can_continue():
    app = create_app(
        llm=ScriptedLLM(),
        settings=Settings(),
        search=FakeSearch(hits=by_filter),
        store=InMemoryStore(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/chat", json={"message": "I want a waterfall hike", "mode": "agent"}
        )
        from test_graph_api import parse_sse

        session_id = parse_sse(first.text)[0][1]["session_id"]
        assert len(session_id) == 12
        second = await client.post(
            "/chat", json={"message": "show me more", "mode": "agent", "session_id": session_id}
        )
    assert done(parse_sse(second.text))["category"] == "more"


def test_a_tie_is_about_closeness_not_order():
    unsorted = [
        hit("a", "A", 0.40, "Easy", 1, 1),
        hit("b", "B", 0.405, "Easy", 2, 2),
        hit("c", "C", 0.60, "Easy", 3, 3),
    ]
    from trailmate_agent.answers import tied_hits

    assert [h.slug for h in tied_hits(unsorted, 0.01)] == [
        "a",
        "b",
    ]  # B is higher but still within the margin of A
