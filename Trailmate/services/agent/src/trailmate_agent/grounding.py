"""Stop the model from stating distances or climbs that are not in the search results.

Trail facts come from the database. The model only writes the explanation, and this check enforces it:
every number written with a distance/height unit must be one of the retrieved facts (or one the user
typed). Anything else means the model made it up.
"""

import re

from trailmate_agent.tools import TrailHit

_UNITS = r"(?:km|kilomet(?:er|re)s?|m|meters?|metres?|miles?|mi|ft|feet)"
_MEASURE = re.compile(rf"(\d+(?:\.\d+)?)\s*{_UNITS}\b", re.IGNORECASE)


def measures_in(text: str) -> list[float]:
    return [float(n) for n in _MEASURE.findall(text)]


def allowed_numbers(hits: list[TrailHit]) -> list[float]:
    nums: list[float] = []
    for hit in hits:
        nums += [hit.facts.distance_km, float(hit.facts.elevation_gain_m)]
    return nums


def unsupported_numbers(reply: str, allowed: list[float]) -> list[str]:
    """Measurements in the reply that are not in `allowed`. Empty means the reply is grounded."""
    return [f"{n:g}" for n in measures_in(reply) if not any(abs(n - a) < 1e-6 for a in allowed)]
