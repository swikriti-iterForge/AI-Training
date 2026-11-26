"""Deterministic routing: plain code decides which path a message takes. No model involved.

Three categories (the Phase 1 spec):
  trail  - a question about hiking trails        -> answered by the LLM
  chat   - greetings, thanks, "what can you do"   -> templated reply
  reject - something we won't or can't do         -> templated reply

Honest limitation (and the reason Phase 3 exists): keyword rules are brittle. "I want to
walk somewhere scenic" works only because "walk" is on the list; "somewhere to clear my
head" would fall through to chat.
"""

import re
from typing import Literal

Category = Literal["trail", "chat", "reject"]

# Checked FIRST: things this app must not do, even when the message also mentions a trail.
REJECT_PATTERNS = [
    # live conditions: we have no live data, so we refuse rather than guess
    r"\b(weather|forecast|temperature|rain(ing)?|snow(ing)?)\b.{0,40}\b(today|tomorrow|tonight|now|this weekend)\b",
    r"\b(today|tomorrow|tonight|now|this weekend)\b.{0,40}\b(weather|forecast|temperature)\b",
    r"\bis (it|the trail|the \w+( \w+)?) (open|closed|safe|flooded|icy|muddy)\b.{0,20}\b(today|now|right now|tomorrow)\b",
    # clearly not about hiking
    r"\b(write|debug|fix)\b.{0,20}\b(code|program|script|essay|poem)\b",
    r"\b(stock|crypto|bitcoin|election|politic\w*|homework|recipe)\b",
]

TRAIL_KEYWORDS = [
    "trail",
    "trails",
    "trailhead",
    "hike",
    "hikes",
    "hiking",
    "hiker",
    "trek",
    "treks",
    "trekking",
    "walk",
    "walks",
    "walking",
    "stroll",
    "scramble",
    "summit",
    "peak",
    "ridge",
    "viewpoint",
    "overlook",
    "waterfall",
    "waterfalls",
    "falls",
    "gorge",
    "loop",
    "route",
    "path",
    "elevation",
    "switchback",
    "switchbacks",
    "creek",
    "lake",
    "meadow",
    "alpine",
    "dog-friendly",
    "backpacking",
    "camping",
    "sunset",
    "sunrise",
]

CHAT_PATTERNS = {
    "greeting": r"^\s*(hi|hello|hey|hiya|namaste|good (morning|afternoon|evening))\b",
    "thanks": r"\b(thanks|thank you|thx|cheers|appreciate it)\b",
    "bye": r"\b(bye|goodbye|see you|see ya|talk later)\b",
    "about": r"\b(who are you|what are you|your name|what can you do|how do you work|help me|what do you do)\b",
}


def classify(message: str) -> Category:
    text = message.lower()
    if any(re.search(p, text) for p in REJECT_PATTERNS):
        return "reject"
    words = set(re.findall(r"[a-z][a-z-]*", text))
    if words & set(TRAIL_KEYWORDS):
        return "trail"
    return "chat"


def chat_kind(message: str) -> str:
    """Which templated chat reply to use. Falls back to 'unknown' (a gentle redirect)."""
    text = message.lower()
    for kind, pattern in CHAT_PATTERNS.items():
        if re.search(pattern, text):
            return kind
    return "unknown"


# ---------- Templated replies (fixed text, fully predictable) ----------
CHAT_REPLIES = {
    "greeting": "Hi! I'm TrailMate, your hiking companion. Tell me what kind of hike you're after: easy or challenging, short or long, waterfalls or views.",
    "thanks": "You're welcome! Happy to help with more trails any time.",
    "bye": "Happy trails! Come back whenever you want to plan another hike.",
    "about": 'I\'m TrailMate. I help you find a hiking trail that fits what you\'re looking for. Try: "an easy trail near water" or "a hard hike with a big view".',
    "unknown": "I'm built to help with hiking trails, so I'm not sure about that one. Tell me what kind of hike you want and I'll help.",
}

REJECT_REPLY = (
    "I can't help with that one. I don't have live information like current weather or "
    "trail conditions, and I stick to hiking trails. Ask me about a trail and I'm glad to help."
)


# ---------- Agent mode (Phase 3) ----------
# Rules still handle the obvious cases (greetings, thanks, weather refusals). The model only sees the rest.

# Words that say nothing about WHAT the user wants. If nothing else is left, the request is too vague.
FILLER_WORDS = set(
    """a an the i me my we us you your want need would like looking look for to of in on at near around
    find show suggest recommend give tell about some any good nice great best cool new another different
    something anything please can could help is are be do does it that this there one ones place places
    spot spots trip outing trail trails trailhead hike hikes hiking hiker trek treks trekking walk walks
    walking route routes path paths option options idea ideas""".split()
)


def triage(message: str) -> Literal["chat", "reject", "agent"]:
    """Who handles this message: a templated reply, a refusal, or the model."""
    text = message.lower()
    if any(re.search(p, text) for p in REJECT_PATTERNS):
        return "reject"
    if set(re.findall(r"[a-z][a-z-]*", text)) & set(TRAIL_KEYWORDS):
        return "agent"  # it is about trails: let the model decide what to do
    if chat_kind(message) != "unknown":
        return "chat"  # greeting, thanks, goodbye, "what can you do"
    return "agent"


def is_vague(message: str) -> bool:
    """True if, once filler words are removed, nothing is left to search for ('Suggest a good trek')."""
    return not [w for w in re.findall(r"[a-z][a-z-]*", message.lower()) if w not in FILLER_WORDS]


def mentions_location(message: str) -> bool:
    """Crude: 'near Mumbai', 'around Pune', 'in Lonavala'. Our data has no locations, so we say so."""
    return bool(
        re.search(r"\b(?:near|around|in|close to|outside)\s+(?:the\s+)?[A-Z][a-zA-Z]+", message)
    )
