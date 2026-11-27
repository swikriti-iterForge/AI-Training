"""Shared test doubles for the agent tests: a scripted model, a fake search service, an in-memory
preference store, and helpers to run one message or a whole conversation through the real API and graph."""

from httpx import ASGITransport, AsyncClient
from test_graph_api import parse_sse

from trailmate_agent.config import Settings
from trailmate_agent.llm import Completion, ToolCall
from trailmate_agent.main import create_app
from trailmate_agent.prefs import Prefs
from trailmate_agent.tools import Facts, SearchResults, SearchUnavailable, TrailHit, TrailInfo


def hit(
    slug, name, score, difficulty, km, climb, best_for=("hiking",), dogs="Yes, on leash", terms=()
):
    return TrailHit(
        slug=slug,
        name=name,
        score=score,
        matched_section="Overview",
        excerpt=f"{name} is a trail.",
        facts=Facts(
            difficulty=difficulty,
            distance_km=km,
            elevation_gain_m=climb,
            best_for=list(best_for),
            dogs_allowed=dogs,
        ),
        terms_found=list(terms),
    )


EXTENSION = hit(
    "cascade-falls-extension",
    "Cascade Falls Extension",
    0.60,
    "Easy-Moderate",
    5.5,
    180,
    terms=("waterfall",),
)
LOOP = hit("cascade-falls-loop", "Cascade Falls Loop", 0.50, "Easy", 3.2, 90, terms=("waterfall",))
GORGE = hit(
    "thunder-gorge-trail",
    "Thunder Gorge Trail",
    0.40,
    "Hard",
    12.5,
    890,
    terms=("waterfall", "scramble"),
)
BLACKROCK = hit(
    "blackrock-summit", "Blackrock Summit", 0.47, "Hard", 11.0, 980, terms=("scramble", "view")
)
CLEAR = [EXTENSION, LOOP, GORGE]  # a clear winner
TIED = [
    hit(
        "cascade-falls-extension",
        "Cascade Falls Extension",
        0.535,
        "Easy-Moderate",
        5.5,
        180,
        terms=("waterfall",),
    ),
    hit("cascade-falls-loop", "Cascade Falls Loop", 0.534, "Easy", 3.2, 90, terms=("waterfall",)),
    GORGE,
]
GOOD_ANSWER = (
    "Cascade Falls Extension suits you: it is Easy-Moderate at 5.5 km with a 180 m climb. ",
    "The Loop is a shorter option at 3.2 km.",
)


class ScriptedLLM:
    """Stands in for the model. `decide` is what complete() returns; `write` is what stream() yields."""

    def __init__(self, decide=None, write=GOOD_ANSWER):
        self.decide = (
            decide
            if decide is not None
            else Completion("", [ToolCall("search_trails", '{"query": "easy waterfall hike"}')])
        )
        self.write = write
        self.complete_calls, self.stream_calls = [], []

    async def complete(self, messages, tier, tools):
        self.complete_calls.append((messages, tier, tools))
        if isinstance(self.decide, Exception):
            raise self.decide
        return self.decide

    async def stream(self, messages, tier):
        self.stream_calls.append((messages, tier))
        if isinstance(self.write, Exception):
            raise self.write
        for piece in self.write:
            yield piece


class FakeSearch:
    """Stands in for the search service. `hits` is a list, or a function (difficulty, exclude) -> list."""

    def __init__(self, hits=CLEAR, fail=False, trails=None):
        self.hits, self.fail = hits, fail
        self.queries, self.calls = [], []
        self.trails = trails or {
            h.slug: TrailInfo(slug=h.slug, name=h.name, tagline=f"{h.name} tagline.", facts=h.facts)
            for h in [*CLEAR, BLACKROCK]
        }

    async def search(self, query, top_k=3, *, mode=None, difficulty=None, exclude=None, terms=None):
        self.queries.append((query, top_k))
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "difficulty": difficulty,
                "exclude": exclude,
                "terms": terms,
            }
        )
        if self.fail:
            raise SearchUnavailable("connection refused")
        hits = self.hits(difficulty, exclude) if callable(self.hits) else self.hits
        return SearchResults(query=query, results=hits)

    async def get_trail(self, slug):
        if self.fail:
            raise SearchUnavailable("connection refused")
        return self.trails[slug]


class InMemoryStore:
    """The PreferenceStore protocol without a file."""

    def __init__(self):
        self.data: dict[str, Prefs] = {}

    def load(self, user_id):
        return self.data.get(user_id, Prefs())

    def save(self, user_id, prefs):
        self.data[user_id] = prefs

    def delete(self, user_id):
        return self.data.pop(user_id, None) is not None


async def agent(llm, search, message, *, llm_writer=True, store=None, settings=None, **extra):
    """One message through agent mode. llm_writer=True lets the model write the answer (the opt-in path)."""
    app = create_app(
        llm=llm,
        settings=settings or Settings(explain_with_llm=llm_writer),
        search=search,
        store=store or InMemoryStore(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/chat", json={"message": message, "mode": "agent", **extra})
    return response, parse_sse(response.text) if response.status_code == 200 else []


class Conversation:
    """Several turns through ONE app, so the session memory is real. Uses the real default settings."""

    def __init__(
        self, llm=None, search=None, *, store=None, settings=None, user_id="u1", session_id="s1"
    ):
        self.llm, self.search = llm or ScriptedLLM(), search or FakeSearch()
        self.store, self.user_id, self.session_id = store or InMemoryStore(), user_id, session_id
        self.app = create_app(
            llm=self.llm, settings=settings or Settings(), search=self.search, store=self.store
        )

    async def say(self, message, *, session_id=None, user_id="__default__", **extra):
        uid = self.user_id if user_id == "__default__" else user_id
        body = {
            "message": message,
            "mode": "agent",
            "session_id": session_id or self.session_id,
            **extra,
        }
        if uid:
            body["user_id"] = uid
        async with AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            response = await client.post("/chat", json=body)
        assert response.status_code == 200, response.text
        return parse_sse(response.text)


def event(events, kind):
    return [data for e, data in events if e == kind]
