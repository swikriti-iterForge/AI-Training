"""Reading what the user wants from plain words, with rules (no model): difficulty, features, follow-up
commands, and place names. Everything here is deterministic and unit-tested.

Why rules and not the model: in testing, the small models over-called tools and echoed the user's words
(including a place the data knows nothing about). Rules are predictable and free, and the model is kept for
the one thing that is genuinely open-ended: what to search for.
"""

import re
from dataclasses import dataclass, replace
from typing import Any

from trailmate_agent.routing import FILLER_WORDS, LOCATION_RE

DIFFICULTY_WORDS = {
    "easy": "Easy",
    "easier": "Easy",
    "gentle": "Easy",
    "beginner": "Easy",
    "beginners": "Easy",
    "relaxed": "Easy",
    "leisurely": "Easy",
    "hard": "Hard",
    "harder": "Hard",
    "difficult": "Hard",
    "challenging": "Hard",
    "strenuous": "Hard",
    "tough": "Hard",
    "demanding": "Hard",
    "moderate": "Moderate",
    "medium": "Moderate",
    "intermediate": "Moderate",
}
# Concrete things a trail's description can mention. Plurals map to the singular the search checks for.
FEATURE_WORDS = {
    "waterfall": "waterfall",
    "waterfalls": "waterfall",
    "view": "view",
    "views": "view",
    "viewpoint": "viewpoint",
    "overlook": "overlook",
    "sunset": "sunset",
    "sunrise": "sunrise",
    "forest": "forest",
    "woods": "woods",
    "lake": "lake",
    "creek": "creek",
    "river": "river",
    "meadow": "meadow",
    "wildflower": "wildflower",
    "wildflowers": "wildflower",
    "alpine": "alpine",
    "snow": "snow",
    "coastal": "coastal",
    "cliff": "cliff",
    "cliffs": "cliff",
    "ruins": "ruins",
    "history": "history",
    "scramble": "scramble",
    "boardwalk": "boardwalk",
}
DOG_WORDS = {"dog", "dogs", "puppy", "dog-friendly"}
# How busy a trail is cannot be checked reliably (a description that says "crowds" may mean "no crowds").
QUIET_WORDS = {
    "quiet",
    "crowd",
    "crowds",
    "crowded",
    "busy",
    "popular",
    "secluded",
    "solitude",
    "peaceful",
}
OPPOSITE = {"Easy": "Hard", "Hard": "Easy"}
# How long a trail is. Every trail has a distance, so "short" and "long" are real filters, not search words.
# The limits come from this data: every Easy trail is <= 4.0 km and every Hard trail is >= 8.2 km, with clear gaps.
LENGTH_WORDS = {
    "short": "short",
    "shorter": "short",
    "quick": "short",
    "brief": "short",
    "long": "long",
    "longer": "long",
    "lengthy": "long",
    "full-day": "long",
    "all-day": "long",
}
LENGTH_OPPOSITE = {"short": "long", "long": "short"}
SHORT_MAX_KM = 4.0
LONG_MIN_KM = 8.0


@dataclass(frozen=True)
class Prefs:
    difficulty: str | None = None
    terms: tuple[str, ...] = ()
    dogs: bool = False
    quiet: bool = False
    length: str | None = None  # short | long

    @property
    def is_empty(self) -> bool:
        return not (self.difficulty or self.terms or self.dogs or self.quiet or self.length)

    def to_dict(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "terms": list(self.terms),
            "dogs": self.dogs,
            "quiet": self.quiet,
            "length": self.length,
        }

    @staticmethod
    def from_dict(data: dict) -> "Prefs":
        return Prefs(
            difficulty=data.get("difficulty") or None,
            terms=tuple(str(t) for t in data.get("terms", [])),
            dogs=bool(data.get("dogs")),
            quiet=bool(data.get("quiet")),
            length=data.get("length") or None,
        )


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]*", text.lower())


def _level(words: list[str], table: dict[str, str], opposite: dict[str, str]) -> str | None:
    """The ONE level these words ask for, or None. 'not too hard' means easy; 'easy or hard' means no preference."""
    found: set[str] = set()
    for i, word in enumerate(words):
        value = table.get(word)
        if value is None:
            continue
        before = words[max(0, i - 3) : i]
        if any(w in {"not", "no", "never"} or w.endswith("n't") for w in before):
            value = opposite.get(value)  # 'not moderate' says nothing useful
        if value:
            found.add(value)
    return next(iter(found)) if len(found) == 1 else None


def extract_prefs(message: str) -> Prefs:
    """Difficulty, length, features and flags in ONE message."""
    words = _words(message)
    return Prefs(
        difficulty=_level(words, DIFFICULTY_WORDS, OPPOSITE),
        terms=tuple(dict.fromkeys(FEATURE_WORDS[w] for w in words if w in FEATURE_WORDS)),
        dogs=any(w in DOG_WORDS for w in words),
        quiet=any(w in QUIET_WORDS for w in words),
        length=_level(words, LENGTH_WORDS, LENGTH_OPPOSITE),
    )


def merge_prefs(base: Prefs, new: Prefs) -> Prefs:
    """New wishes override old ones. A new topic REPLACES the old features (so a waterfall chat does not
    leak into an unrelated 'alpine' question); 'something harder' keeps them. Flags stick."""
    return Prefs(
        difficulty=new.difficulty or base.difficulty,
        terms=new.terms or base.terms,
        dogs=base.dogs or new.dogs,
        quiet=base.quiet or new.quiet,
        length=new.length or base.length,
    )


def minus(a: Prefs, b: Prefs) -> Prefs:
    """What `a` contributes that `b` did not already say (used to tell the user what we reused)."""
    return Prefs(
        difficulty=a.difficulty if a.difficulty and not b.difficulty else None,
        terms=a.terms if a.terms and not b.terms else (),
        dogs=a.dogs and not b.dogs,
        quiet=a.quiet and not b.quiet,
        length=a.length if a.length and not b.length else None,
    )


def describe(p: Prefs) -> str:
    parts = []
    if p.difficulty:
        parts.append(f"{p.difficulty.lower()} trails")
    if p.length:
        parts.append(f"{p.length} trails")
    parts += list(p.terms)
    if p.dogs:
        parts.append("dog-friendly")
    if p.quiet:
        parts.append("quiet")
    return ", ".join(parts)


# "I love waterfalls", "I always prefer easy trails": a standing preference, not a one-off request.
# "I want an easy hike" is only a request, so it is counted, not assumed.
_STANDING = re.compile(
    r"\bi (?:really |always |usually |generally |mostly )?(?:prefer|like|love|enjoy)\b|\bi always\b|\bmy (?:favou?rite|usual)\b",
    re.IGNORECASE,
)


def states_preference(message: str) -> bool:
    return bool(_STANDING.search(message)) and "?" not in message


def is_filter_only(message: str, difficulty: str | None = None, length: str | None = None) -> bool:
    """'okay show some easy ones' / 'short and challenging': it says HOW HARD or HOW LONG, but nothing about WHAT
    to see. The filters ARE the request, and a content-free query scores low against everything, so a low
    similarity must not mean "refuse"."""
    if not (difficulty or length):
        return False
    return only_preference_words(message)


def only_preference_words(message: str) -> bool:
    """True if, besides filler, the message uses only words we can actually filter by (difficulty, length,
    dogs) or that we can never verify anyway ('quiet', 'peaceful': not a data field). Such a message must
    never be refused for a low search score, because there is nothing left for the search to score well on."""
    words = re.findall(r"[a-z][a-z'-]*", message.lower())
    harmless = (
        FILLER_WORDS | DIFFICULTY_WORDS.keys() | LENGTH_WORDS.keys() | DOG_WORDS | QUIET_WORDS
    )
    return not [w for w in words if w not in harmless]


def without(p: Prefs, *fields: str) -> Prefs:
    """`p` with the named fields cleared back to their default (used to suppress a dropped preference)."""
    defaults: dict[str, Any] = {
        "difficulty": None,
        "length": None,
        "terms": (),
        "dogs": False,
        "quiet": False,
    }
    changes = {f: defaults[f] for f in fields}
    return replace(p, **changes)  # type: ignore[arg-type]


# ---------- follow-up commands ----------
_MORE = re.compile(
    r"(?:please )?(?:(?:show|give|get|see) )?(?:me )?(?:(?:some|any) )?(?:more|other|another|different)"
    r"(?: (?:options?|trails?|hikes?|ones?|suggestions?|results?))?(?: please)?"
    r"|(?:something|anything|what) else(?: please)?"
)
_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}


def follow_up(message: str, has_shown: bool) -> tuple[str, int | None] | None:
    """Commands about the conversation itself. -> ("forget"|"recall"|"remember"|"reset"|"more"|"pick", n)."""
    text = " ".join(re.sub(r"[^a-z0-9#' ]+", " ", message.lower()).split())
    if re.search(
        r"\b(forget|delete|erase|clear)\b.*\b(preferences?|memory|what you (?:know|remember))\b",
        text,
    ):
        return ("forget", None)
    if re.search(
        r"\bwhat (?:do|did) you (?:remember|know) about me\b|\b(?:my|saved) preferences\b|\bwhat have you saved\b"
        r"|\bdo you (?:know|remember) what (?:kind of |type of |sort of )?(?:trails?|hikes?|treks?|things?)? ?i (?:like|prefer|love|enjoy)\b"
        r"|\bwhat (?:kind of |type of )?(?:trails?|hikes?|treks?) do i (?:like|prefer|love|enjoy)\b"
        r"|\bdo you (?:know|remember) me\b",
        text,
    ):
        return ("recall", None)
    if re.match(r"^(?:please )?remember\b", text) or re.search(
        r"\bremember (?:that )?(?:i|my)\b", text
    ):
        return ("remember", None)
    if re.search(r"\b(?:start over|start again|reset|new search|never ?mind)\b", text):
        return ("reset", None)
    if has_shown and len(text.split()) <= 8:
        ordinal = re.search(
            r"\b(first(?! time)|1st|second|2nd|third|3rd)\b|(?:#\s?|\b(?:number|option|trail)\s+)([123])\b",
            text,
        )
        if ordinal and re.search(r"\b(one|option|trail|number|about|details?|tell)\b|#", text):
            return (
                "pick",
                _ORDINALS[ordinal.group(1)] if ordinal.group(1) else int(ordinal.group(2)),
            )
        if _MORE.fullmatch(text):
            return ("more", None)
    return None


# ---------- place names ----------
def strip_location(query: str, message: str) -> str:
    """Remove the places the USER named from the search query. The data has no locations, and in testing
    'near Mumbai' lowered the top score from 0.640 to 0.529 and brought in a Hard trail for an 'easy' request."""
    cleaned = query
    for match in LOCATION_RE.finditer(message):
        phrase = match.group(0)
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        place = re.sub(
            r"^(?:near|around|in|close to|outside)\s+(?:the\s+)?", "", phrase, flags=re.IGNORECASE
        )
        cleaned = re.sub(rf"\b{re.escape(place)}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:near|around|close to|outside)\s*(?=[?.!,]|$)", " ", cleaned, flags=re.IGNORECASE
    )
    return " ".join(cleaned.split()) or query


def place_names(message: str) -> list[str]:
    """The place words the user typed, e.g. ['Mumbai']. Used to catch a reply that echoes them."""
    names = []
    for match in LOCATION_RE.finditer(message):
        place = re.sub(
            r"^(?:near|around|in|close to|outside)\s+(?:the\s+)?",
            "",
            match.group(0),
            flags=re.IGNORECASE,
        )
        names += [w for w in place.split() if len(w) > 2]
    return names