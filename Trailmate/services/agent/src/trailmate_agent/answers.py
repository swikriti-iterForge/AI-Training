"""Replies written by code, not by the model: fixed text built only from retrieved facts.

Used when the request is too vague (ask), when two trails are nearly tied (compare them honestly),
and as the fallback when the model cannot write an explanation.
"""

from trailmate_agent.prefs import Prefs, describe
from trailmate_agent.tools import TrailHit, TrailInfo

CLARIFY_QUESTION = (
    "Happy to help you find a trail! What matters most: how hard it should be (easy or challenging), "
    "how long it should be, or what you'd like to see (a waterfall, a view, forest)?"
)
LOCATION_NOTE = "Note: my trail data doesn't include locations, so I can't filter by where you are."
SEARCH_UNAVAILABLE = "I can't reach the trail search right now, so I can't recommend a trail. Please try again in a moment."
NO_RESULTS = "I couldn't find a trail that matches that. Could you describe what you're after a little differently?"
NO_MATCH = (
    "I couldn't find a trail that matches that closely, and I'd rather not guess. "
    "What do you enjoy: easy or hard hikes, waterfalls, views, forest?"
)
NO_MORE = "That's everything I have for this search. Want to change something, like the difficulty or what you'd like to see?"
RELAXED_NOTE = "Note: I found no {difficulty} trails for this, so these are the closest matches of any difficulty."
REMEMBER_HELP = 'I can remember things like "I prefer easy trails" or "I love waterfalls". Try: remember that I prefer easy trails.'
NEED_USER = "I can't save preferences without a user id, so nothing was saved. (The chat client sends one; see --user.)"
RESET_REPLY = "Okay, starting fresh. What kind of trail are you after?"
FORGET_REPLY = (
    "Done. I've forgotten what I learned about you and what you told me in this chat. "
    "Your saved chats stay until you delete them from the history list."
)


def facts_line(hit: TrailHit) -> str:
    f = hit.facts
    return f"{f.difficulty}, {f.distance_km:g} km, {f.elevation_gain_m} m climb"


def detail_line(hit: TrailHit) -> str:
    return f"{hit.name}: {facts_line(hit)}. Best for: {', '.join(hit.facts.best_for)}. Dogs: {hit.facts.dogs_allowed}."


def tied_hits(hits: list[TrailHit], margin: float) -> list[TrailHit]:
    """The top results whose score is within `margin` of the best one. Empty unless at least two tie."""
    if len(hits) < 2:
        return []
    top = hits[0].score
    tied = [h for h in hits if round(abs(top - h.score), 4) < margin]
    return tied if len(tied) >= 2 else []


def disambiguation_reply(tied: list[TrailHit]) -> str:
    """Don't pretend to know which of nearly-equal trails is best: show both and ask."""
    if len(tied) == 2:
        a, b = tied
        diffs = []
        if a.facts.difficulty != b.facts.difficulty:
            diffs.append(f"{a.name} is {a.facts.difficulty} and {b.name} is {b.facts.difficulty}")
        if a.facts.distance_km != b.facts.distance_km:
            longer, shorter = (a, b) if a.facts.distance_km > b.facts.distance_km else (b, a)
            diffs.append(
                f"{longer.name} is longer ({longer.facts.distance_km:g} km against {shorter.facts.distance_km:g} km)"
            )
        if a.facts.elevation_gain_m != b.facts.elevation_gain_m:
            steeper, flatter = (
                (a, b) if a.facts.elevation_gain_m > b.facts.elevation_gain_m else (b, a)
            )
            diffs.append(
                f"{steeper.name} has more climbing ({steeper.facts.elevation_gain_m} m against {flatter.facts.elevation_gain_m} m)"
            )
        header = "Two trails fit your request almost equally well, so I won't pick one for you:"
        differences = (
            ("Main differences: " + "; ".join(diffs) + ".")
            if diffs
            else "They look alike on the facts I have."
        )
    else:
        header = "Several trails fit your request about equally well:"
        differences = "They differ in difficulty, length and climb, as listed above."
    lines = "\n".join(f"- {detail_line(h)}" for h in tied)
    return f"{header}\n{lines}\n{differences} Tell me which sounds closer to what you want and I'll narrow it down."


def fit(hit: TrailHit, prefs: Prefs) -> tuple[list[str], list[tuple[str, str]]]:
    """What a trail matches and misses, judged ONLY from its facts and the words its description contains.
    -> (matched statements, [(short label, explanation)] for what it misses)."""
    matched: list[str] = []
    missed: list[tuple[str, str]] = []
    if prefs.difficulty:
        label = prefs.difficulty.lower()
        if hit.facts.difficulty == prefs.difficulty:
            matched.append(f"{label}: it is rated {hit.facts.difficulty}")
        else:
            missed.append((label, f"{label} (it is rated {hit.facts.difficulty})"))
    for term in prefs.terms:
        if term in hit.terms_found:
            matched.append(f"{term}: its description mentions it")
        else:
            missed.append((term, f"{term} (its description doesn't mention it)"))
    return matched, missed


def explain_reply(hits: list[TrailHit], prefs: Prefs) -> str:
    """The recommendation, built by code from retrieved facts. It says why a trail fits, what it does NOT
    match, and what the data cannot tell us. Nothing here comes from a model, so nothing can be invented."""
    best = hits[0]
    matched, missed = fit(best, prefs)
    lines = [f"Best match: {best.name} ({facts_line(best)})."]
    lines.append(
        "Why it fits: "
        + ("; ".join(matched) if matched else "it is the closest match to what you described")
        + "."
    )
    if missed:
        lines.append("Not matched: " + "; ".join(detail for _, detail in missed) + ".")
    if prefs.dogs:
        lines.append(f"Dogs: {best.facts.dogs_allowed}.")
    if prefs.quiet:
        lines.append(
            "Not in my data: how busy a trail is. I can only point to what each description says."
        )
    if len(hits) > 1:
        others = []
        for hit in hits[1:3]:
            _, hit_missed = fit(hit, prefs)
            extra = (
                ("; not matched: " + ", ".join(label for label, _ in hit_missed))
                if hit_missed
                else ""
            )
            others.append(f"{hit.name} ({facts_line(hit)}{extra})")
        lines.append("Also worth a look: " + "; ".join(others) + ".")
    return "\n".join(lines)


def details_reply(info: TrailInfo) -> str:
    f = info.facts
    head = f"{info.name}: {info.tagline}" if info.tagline else info.name
    return f"{head}\n{f.difficulty}, {f.distance_km:g} km, {f.elevation_gain_m} m climb. Best for: {', '.join(f.best_for)}. Dogs: {f.dogs_allowed}."


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def asked_for(learned: dict | None) -> str:
    """'easy trails (3 times), waterfall (2 times)': what this user has asked for so far."""
    if not learned:
        return ""
    items = [(f"{d.lower()} trails", n) for d, n in (learned.get("difficulty") or {}).items()]
    items += list((learned.get("terms") or {}).items())
    for key, label in (("dogs", "dog-friendly"), ("quiet", "quiet")):
        if learned.get(key):
            items.append((label, learned[key]))
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name} ({_times(n)})" for name, n in items[:6])


def remembered_reply(known: Prefs, session: Prefs, learned: dict | None = None) -> str:
    parts = []
    if not known.is_empty:
        parts.append(f"What I remember about you: {describe(known)}.")
    seen = asked_for(learned)
    if seen:
        parts.append(f"So far you've asked for: {seen}.")
    if not session.is_empty:
        parts.append(f"In this chat: {describe(session)}.")
    return (
        " ".join(parts)
        if parts
        else "I'm not remembering anything about you yet. I learn from what you ask for."
    )
