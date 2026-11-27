"""Automatic learning, "what do you know about me", saved chats, and reopening a chat after a restart.

Same style as the other agent tests: a scripted model, a fake search service, the real API and graph, and the
REAL stores writing to throwaway folders."""

import json

import pytest
from agent_fakes import (
    Conversation,
    FakeSearch,
    InMemoryStore,
    event,
    tmp_conversations,
)
from httpx import ASGITransport, AsyncClient
from test_agent_memory import by_filter
from test_graph_api import done, names, streamed_text

from trailmate_agent.answers import asked_for
from trailmate_agent.config import Settings
from trailmate_agent.conversations import FileConversationStore
from trailmate_agent.ids import valid_id
from trailmate_agent.main import create_app
from trailmate_agent.prefs import Prefs, follow_up, states_preference
from trailmate_agent.store import FilePreferenceStore

# ============ the preference store: counting what was asked ============


def test_one_request_is_counted_but_not_assumed(tmp_path):
    store = FilePreferenceStore(tmp_path / "p.json", min_mentions=2)
    store.learn("a", Prefs(difficulty="Hard", terms=("waterfall",)))
    assert store.load("a").is_empty  # a one-off ("I wanted a hard hike once") is not a preference
    assert (
        store.summary("a")["learned"]["difficulty"] == {"Hard": 1}
        and store.summary("a")["learned"]["requests"] == 1
    )


def test_asked_twice_becomes_a_usual_preference(tmp_path):
    store = FilePreferenceStore(tmp_path / "p.json", min_mentions=2)
    for _ in range(2):
        store.learn("a", Prefs(difficulty="Easy", terms=("waterfall",), dogs=True))
    assert store.load("a") == Prefs(difficulty="Easy", terms=("waterfall",), dogs=True)
    assert store.summary("a")["explicit"] == Prefs().to_dict()  # observed, not stated: kept apart


def test_a_coin_toss_between_difficulties_is_no_preference(tmp_path):
    store = FilePreferenceStore(tmp_path / "p.json", min_mentions=2)
    for level in ("Easy", "Hard", "Easy", "Hard"):
        store.learn("a", Prefs(difficulty=level))
    assert store.load("a").difficulty is None
    store.learn("a", Prefs(difficulty="Easy"))  # now a clear favourite
    assert store.load("a").difficulty == "Easy"


def test_a_standing_statement_counts_immediately_and_wins_over_habit(tmp_path):
    store = FilePreferenceStore(tmp_path / "p.json", min_mentions=2)
    store.learn("a", Prefs(terms=("waterfall",)), strong=True)  # "I love waterfalls"
    assert store.load("a").terms == ("waterfall",) and store.summary("a")["explicit"]["terms"] == [
        "waterfall"
    ]
    for _ in range(3):
        store.learn(
            "a", Prefs(terms=("lake",))
        )  # a habit of asking for lakes does not override what they said
    assert store.load("a").terms == ("waterfall",)


def test_only_the_three_most_asked_features_are_usual(tmp_path):
    store = FilePreferenceStore(tmp_path / "p.json", min_mentions=1)
    store.learn("a", Prefs(terms=("lake", "creek", "forest", "meadow")))
    assert len(store.load("a").terms) == 3


def test_entries_saved_by_the_old_explicit_only_store_still_load(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {"me": {"difficulty": "Easy", "terms": ["lake"], "dogs": False, "quiet": False}}
        ),
        encoding="utf-8",
    )
    assert FilePreferenceStore(path).load("me") == Prefs(difficulty="Easy", terms=("lake",))


def test_the_store_ignores_ids_that_are_not_safe(tmp_path):
    store = FilePreferenceStore(tmp_path / "p.json")
    store.learn("../evil", Prefs(difficulty="Easy"), strong=True)
    assert store.load("../evil").is_empty and not (tmp_path / "p.json").exists()


@pytest.mark.parametrize(
    "value, ok",
    [
        ("alice", True),
        ("A1_b.c-d", True),
        ("..", False),
        (".hidden", False),
        ("a/b", False),
        ("", False),
        ("x" * 65, False),
        (None, False),
    ],
)
def test_ids(value, ok):
    assert valid_id(value) is ok


def test_asked_for_reads_naturally():
    learned = {
        "difficulty": {"Easy": 3, "Hard": 1},
        "terms": {"waterfall": 4},
        "dogs": 0,
        "quiet": 2,
    }
    assert (
        asked_for(learned)
        == "waterfall (4 times), easy trails (3 times), quiet (2 times), hard trails (once)"
    )
    assert asked_for(None) == "" and asked_for({"difficulty": {}, "terms": {}}) == ""


@pytest.mark.parametrize(
    "message, strong",
    [
        ("I love waterfalls", True),
        ("I always prefer easy trails", True),
        ("my favourite is a quiet forest", True),
        ("I want an easy hike", False),
        ("do you know what trails I like?", False),
        ("Hi", False),
    ],
)
def test_standing_statements(message, strong):
    assert states_preference(message) is strong


@pytest.mark.parametrize(
    "message",
    [
        "do you know what kind of trails i like?",
        "what do you know about me",
        "what trails do i like",
        "do you remember me",
        "show my preferences",
    ],
)
def test_recall_questions_are_recognised(message):
    assert follow_up(message, has_shown=False) == ("recall", None)


# ============ learning inside a conversation ============
async def test_it_learns_without_being_told_and_uses_it_in_a_new_chat():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want an easy waterfall hike")  # asked once: counted only
    assert c.store.load("u1").is_empty
    await c.say(
        "another easy waterfall hike please", session_id="monday"
    )  # asked twice: now a usual preference
    assert c.store.load("u1") == Prefs(difficulty="Easy", terms=("waterfall",))

    events = await c.say("Suggest a good trek", session_id="a-new-chat")  # vague, but it knows
    assert done(events)["decision"] != "clarify"
    assert c.search.calls[-1]["difficulty"] == ["Easy"] and c.search.calls[-1]["terms"] == [
        "waterfall"
    ]
    assert "(Using what I remember about you: easy trails, waterfall.)" in streamed_text(events)


async def test_a_single_request_does_not_change_the_defaults():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want a hard scramble")
    events = await c.say("Suggest a good trek", session_id="other")
    assert (
        streamed_text(events) != "" and done(events)["decision"] == "clarify"
    )  # one-off: still asks


async def test_a_standing_statement_is_learned_at_once():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I love waterfalls")
    assert c.store.load("u1").terms == ("waterfall",)


async def test_what_do_you_know_about_me_answers_from_what_it_learned():
    c = Conversation(search=FakeSearch(hits=by_filter))
    assert "not remembering anything" in streamed_text(
        await c.say("do you know what kind of trails i like?")
    )
    await c.say("I want an easy waterfall hike")
    await c.say("an easy lake walk", session_id="two")
    reply = streamed_text(
        await c.say("do you know what kind of trails i like?", session_id="three")
    )
    assert "What I remember about you: easy trails." in reply
    assert "So far you've asked for: easy trails (2 times)" in reply and "waterfall (once)" in reply


async def test_learning_can_be_switched_off():
    c = Conversation(
        search=FakeSearch(hits=by_filter), settings=Settings(learn_automatically=False)
    )
    await c.say("I want an easy waterfall hike")
    await c.say("another easy waterfall hike", session_id="two")
    assert c.store.data == {}


async def test_without_a_user_nothing_is_learned_or_saved():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want an easy waterfall hike", user_id=None)
    await c.say("I want an easy waterfall hike", user_id=None, session_id="two")
    assert c.store.data == {} and c.conversations.list_chats("u1") == []


async def test_forget_clears_what_was_learned_but_not_the_saved_chats():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want an easy waterfall hike")
    await c.say("another easy waterfall hike", session_id="two")
    assert not c.store.load("u1").is_empty
    await c.say("forget my preferences", session_id="three")
    assert c.store.load("u1").is_empty and c.store.summary("u1")["learned"]["requests"] == 0
    assert len(c.conversations.list_chats("u1")) == 3  # the chats are separate, and still there


# ============ saved chats ============
async def test_each_turn_is_saved_with_its_trail_cards():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want a waterfall hike")
    await c.say("Hello!")
    chat = c.conversations.get("u1", "s1")
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant", "user", "assistant"]
    assert (
        chat["title"] == "I want a waterfall hike"
        and chat["messages"][0]["text"] == "I want a waterfall hike"
    )
    assert (
        chat["messages"][1]["text"].startswith("Best match:")
        and len(chat["messages"][1]["trails"]) == 3
    )
    assert (
        chat["messages"][3]["trails"] is None and chat["memory"]["last_shown"]
    )  # no cards for a greeting


async def test_the_saved_chat_is_there_before_the_client_gets_done():
    """The UI refreshes its history list when 'done' arrives, so the file must already exist."""
    c = Conversation()
    events = await c.say("I want a waterfall hike")
    assert names(events)[-1] == "done" and c.conversations.get("u1", "s1") is not None


async def test_failed_replies_are_saved_too_so_the_history_matches_what_the_user_saw():
    c = Conversation(search=FakeSearch(fail=True))
    await c.say("I want a waterfall hike")
    assert "can't reach the trail search" in c.conversations.get("u1", "s1")["messages"][1]["text"]


async def test_reopening_a_chat_after_a_restart_restores_what_was_shown():
    c = Conversation(search=FakeSearch(hits=by_filter))
    first = await c.say("I want a waterfall hike")
    shown = [t["slug"] for t in event(first, "results")[0]["trails"]]
    c.app = c.new_app()  # the server restarts: the session's memory in RAM is gone
    again = await c.say("show me more")  # same user, same session id
    assert done(again)["category"] == "more"  # it still knows what was shown
    assert c.search.calls[-1]["exclude"] == shown
    assert event(again, "results")[0]["trails"][0]["slug"] not in shown


async def test_a_restart_without_a_saved_chat_just_starts_fresh():
    c = Conversation(search=FakeSearch(hits=by_filter))
    c.app = c.new_app()
    assert done(await c.say("show me more", session_id="never-seen"))["category"] != "more"


async def test_one_user_cannot_continue_or_read_another_users_chat():
    """No restart here on purpose: the session is alive in RAM, which is where a leak between users would show."""
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want a waterfall hike", user_id="alice", session_id="alices-chat")
    stolen = await c.say(
        "show me more", user_id="bob", session_id="alices-chat"
    )  # bob guesses alice's session id
    assert done(stolen)["category"] != "more"  # he does not inherit what alice was shown
    assert c.search.calls[-1]["exclude"] in (None, [])
    assert (
        c.conversations.get("alice", "alices-chat")["messages"][0]["text"]
        == "I want a waterfall hike"
    )
    assert (
        c.conversations.get("bob", "alices-chat")["messages"][0]["text"] == "show me more"
    )  # bob's own file, not alice's


# ============ the conversation store ============
def test_conversation_store_lists_newest_first_and_deletes(tmp_path):
    store = FileConversationStore(tmp_path)
    store.append_turn("a", "one", "first question", "first answer", None, {"seen": []})
    store.append_turn("a", "two", "second question", "second answer", [{"slug": "x"}], None)
    store.append_turn(
        "a", "one", "more", "more answer", None, None
    )  # touches chat one again, so it is the newest
    listing = store.list_chats("a")
    assert (
        [c["id"] for c in listing] == ["one", "two"]
        and listing[0]["messages"] == 4
        and listing[1]["title"] == "second question"
    )
    assert store.get("a", "one")["memory"] == {
        "seen": []
    }  # an update without memory keeps the old one
    assert store.delete("a", "one") is True and store.delete("a", "one") is False
    assert (
        [c["id"] for c in store.list_chats("a")] == ["two"]
        and store.delete_all("a") == 1
        and store.list_chats("a") == []
    )


def test_conversation_store_caps_messages_and_survives_corrupt_files(tmp_path):
    store = FileConversationStore(tmp_path)
    for i in range(250):
        store.append_turn("a", "big", f"q{i}", f"a{i}", None, None)
    chat = store.get("a", "big")
    assert (
        len(chat["messages"]) == 400
        and chat["messages"][-1]["text"] == "a249"
        and chat["title"] == "q0"
    )
    (tmp_path / "a" / "broken.json").write_text("{nope", encoding="utf-8")
    assert [c["id"] for c in store.list_chats("a")] == ["big"] and store.get("a", "broken") is None


@pytest.mark.parametrize(
    "user, session", [("..", "x"), ("a", ".."), ("a/b", "x"), (".", "x"), ("a", "../../etc")]
)
def test_conversation_store_never_leaves_its_folder(tmp_path, user, session):
    root = tmp_path / "chats"
    store = FileConversationStore(root)
    store.append_turn(user, session, "q", "a", None, None)
    assert (
        store.get(user, session) is None
        and store.list_chats(user) == []
        and not list(tmp_path.rglob("*.json"))
    )


# ============ the HTTP surface ============
async def test_conversation_and_memory_endpoints():
    c = Conversation(search=FakeSearch(hits=by_filter))
    await c.say("I want an easy waterfall hike")
    await c.say("another easy waterfall hike", session_id="second")
    async with AsyncClient(transport=ASGITransport(app=c.app), base_url="http://test") as client:
        listing = (await client.get("/conversations/u1")).json()
        assert {x["id"] for x in listing} == {"s1", "second"} and listing[0]["id"] == "second"
        chat = (await client.get("/conversations/u1/s1")).json()
        assert (
            chat["messages"][0]["text"] == "I want an easy waterfall hike"
            and chat["messages"][1]["trails"]
        )
        assert (await client.get("/conversations/u1/nope")).status_code == 404
        assert (await client.get("/conversations/someone-else")).json() == []

        memory = (await client.get("/memory/u1")).json()
        assert memory["known"]["difficulty"] == "Easy" and memory["learned"]["terms"] == {
            "waterfall": 2
        }
        assert memory["known_text"] == "easy trails, waterfall"
        assert memory["asked_text"] == "easy trails (2 times), waterfall (2 times)"

        assert (await client.delete("/conversations/u1/s1")).json() == {"deleted": True}
        assert (await client.delete("/conversations/u1/s1")).json() == {"deleted": False}
        assert (await client.delete("/conversations/u1")).json() == {"deleted": 1}
        assert (await client.get("/conversations/u1")).json() == []
        assert (await client.delete("/memory/u1")).json() == {"deleted": True}
        assert (await client.get("/memory/u1")).json()["known"]["difficulty"] is None


@pytest.mark.parametrize(
    "path",
    [
        "/conversations/.hidden",
        "/memory/.hidden",
        "/conversations/u1/..",
        f"/conversations/{'x' * 65}",
    ],
)
async def test_unsafe_ids_are_refused_by_the_api(path):
    app = create_app(
        llm=None,
        settings=Settings(),
        search=FakeSearch(),
        store=InMemoryStore(),
        conversations=tmp_conversations(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get(path)).status_code in (404, 422)


@pytest.mark.parametrize(
    "field, value", [("user_id", ".."), ("user_id", ".hidden"), ("session_id", "..")]
)
async def test_unsafe_ids_in_a_chat_request_are_refused(field, value):
    app = create_app(
        llm=None,
        settings=Settings(),
        search=FakeSearch(),
        store=InMemoryStore(),
        conversations=tmp_conversations(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (
            await client.post("/chat", json={"message": "hi", "mode": "agent", field: value})
        ).status_code == 422
