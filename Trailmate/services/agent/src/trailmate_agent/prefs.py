"""Reading what the user wants from plain words, with rules (no model): difficulty, features, follow-up
commands, and place names. Everything here is deterministic and unit-tested.

Why rules and not the model: in testing, the small models over-called tools and echoed the user's words
(including a place the data knows nothing about). Rules are predictable and free, and the model is kept for
the one thing that is genuinely open-ended: what to search for.
"""

import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Prefs:
    difficulty: str | None = None
    terms: tuple[str, ...] = ()
    dogs: bool = False
    quiet: bool = False

    @property
    def is_empty(self) -> bool:
        return not (self.difficulty or self.terms or self.dogs or self.quiet)

    def to_dict(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "terms": list(self.terms),
            "dogs": self.dogs,
            "quiet": self.quiet,
        }

    @staticmethod
    def from_dict(data: dict) -> "Prefs":
        return Prefs(
            difficulty=data.get("difficulty") or None,
            terms=tuple(str(t) for t in data.get("terms", [])),
            dogs=bool(data.get("dogs")),
            quiet=bool(data.get("quiet")),
        )


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]*", text.lower())


def extract_prefs(message: str) -> Prefs:
    """Difficulty, features and flags in ONE message. 'not too hard' means easy; 'not easy' means hard."""
    words = _words(message)
    found: set[str] = set()
    for i, word in enumerate(words):
        level = DIFFICULTY_WORDS.get(word)
        if level is None:
            continue
        before = words[max(0, i - 3) : i]
        negated = any(w in {"not", "no", "never"} or w.endswith("n't") for w in before)
        if negated:
            level = OPPOSITE.get(level)  # 'not moderate' says nothing useful
        if level:
            found.add(level)
    terms = tuple(dict.fromkeys(FEATURE_WORDS[w] for w in words if w in FEATURE_WORDS))
    return Prefs(
        difficulty=next(iter(found))
        if len(found) == 1
        else None,  # 'easy or hard' is no preference
        terms=terms,
        dogs=any(w in DOG_WORDS for w in words),
        quiet=any(w in QUIET_WORDS for w in words),
    )


def merge_prefs(base: Prefs, new: Prefs) -> Prefs:
    """New wishes override old ones. A new topic REPLACES the old features (so a waterfall chat does not
    leak into an unrelated 'alpine' question); 'something harder' keeps them. Flags stick."""
    return Prefs(
        difficulty=new.difficulty or base.difficulty,
        terms=new.terms or base.terms,
        dogs=base.dogs or new.dogs,
        quiet=base.quiet or new.quiet,
    )


def minus(a: Prefs, b: Prefs) -> Prefs:
    """What `a` contributes that `b` did not already say (used to tell the user what we reused)."""
    return Prefs(
        difficulty=a.difficulty if a.difficulty and not b.difficulty else None,
        terms=a.terms if a.terms and not b.terms else (),
        dogs=a.dogs and not b.dogs,
        quiet=a.quiet and not b.quiet,
    )


def describe(p: Prefs) -> str:
    parts = []
    if p.difficulty:
        parts.append(f"{p.difficulty.lower()} trails")
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


def is_filter_only(message: str, difficulty: str | None) -> bool:
    """'okay show some easy ones': it says HOW HARD, but nothing about WHAT to see. The difficulty filter IS the
    request, and a content-free query scores low against everything, so a low similarity must not mean "refuse"."""
    if not difficulty:
        return False
    words = re.findall(r"[a-z][a-z'-]*", message.lower())
    return not [w for w in words if w not in FILLER_WORDS and w not in DIFFICULTY_WORDS]


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
