"""Long-term memory: what the agent knows about a user, between conversations and restarts.

It learns in two ways, and both are visible and erasable:
  * explicit  - the user says "remember that I prefer easy trails" or "I love waterfalls": saved at once
  * observed  - every search request is counted (easy 3 times, waterfall 4 times). Something asked for at least
                `min_mentions` times counts as a USUAL preference and is applied when a request doesn't say
                otherwise. One-off requests are only counted, never assumed ("I wanted a hard hike once").

A plain JSON file, one entry per user id, so you can open it and see exactly what is kept. Behind a Protocol, so a
database could replace it later. Not authenticated: anyone who knows a user id can read that user's entry.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol

from trailmate_agent.ids import valid_id
from trailmate_agent.prefs import Prefs, merge_prefs


class PreferenceStore(Protocol):
    def load(self, user_id: str) -> Prefs: ...  # what applies by default: explicit + usual
    def save(self, user_id: str, prefs: Prefs) -> None: ...  # explicit preferences
    def learn(self, user_id: str, turn: Prefs, strong: bool = False) -> None: ...
    def summary(self, user_id: str) -> dict[str, Any]: ...  # {"known", "explicit", "learned"}
    def delete(self, user_id: str) -> bool: ...


def _empty_counts() -> dict[str, Any]:
    return {"requests": 0, "difficulty": {}, "terms": {}, "dogs": 0, "quiet": 0}


class FilePreferenceStore:
    def __init__(self, path: Path, min_mentions: int = 2) -> None:
        self._path = path
        self._min = max(1, min_mentions)
        self._lock = threading.Lock()

    # ---- file handling ----
    def _read(self) -> dict[str, dict]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)  # a crash mid-write can never leave a half-written file

    # ---- what is "known" ----
    def _known(self, entry: dict) -> Prefs:
        explicit = Prefs.from_dict(entry)
        counts = {**_empty_counts(), **entry.get("counts", {})}
        diff = counts["difficulty"]
        usual_difficulty = None
        if diff:
            best, top = max(diff.items(), key=lambda kv: kv[1])
            if (
                top >= self._min and top > sum(diff.values()) - top
            ):  # a clear favourite, not a coin toss
                usual_difficulty = best
        usual_terms = tuple(
            t
            for t, n in sorted(counts["terms"].items(), key=lambda kv: (-kv[1], kv[0]))
            if n >= self._min
        )[:3]
        return Prefs(
            difficulty=explicit.difficulty or usual_difficulty,
            terms=explicit.terms or usual_terms,
            dogs=explicit.dogs or counts["dogs"] >= self._min,
            quiet=explicit.quiet or counts["quiet"] >= self._min,
        )

    # ---- the store interface ----
    def load(self, user_id: str) -> Prefs:
        if not valid_id(user_id):
            return Prefs()
        with self._lock:
            return self._known(self._read().get(user_id, {}))

    def save(self, user_id: str, prefs: Prefs) -> None:
        if not valid_id(user_id):
            return
        with self._lock:
            data = self._read()
            entry = data.get(user_id, {})
            data[user_id] = {**entry, **prefs.to_dict()}
            self._write(data)

    def learn(self, user_id: str, turn: Prefs, strong: bool = False) -> None:
        """Count what one request asked for. `strong` (an "I love..." / "I prefer..." statement) also makes it explicit."""
        if not valid_id(user_id) or turn.is_empty:
            return
        with self._lock:
            data = self._read()
            entry = data.get(user_id, {})
            counts = {**_empty_counts(), **entry.get("counts", {})}
            counts["requests"] += 1
            if turn.difficulty:
                counts["difficulty"][turn.difficulty] = (
                    counts["difficulty"].get(turn.difficulty, 0) + 1
                )
            for term in turn.terms:
                counts["terms"][term] = counts["terms"].get(term, 0) + 1
            counts["dogs"] += int(turn.dogs)
            counts["quiet"] += int(turn.quiet)
            entry["counts"] = counts
            if strong:
                entry.update(merge_prefs(Prefs.from_dict(entry), turn).to_dict())
            data[user_id] = entry
            self._write(data)

    def summary(self, user_id: str) -> dict[str, Any]:
        empty = {
            "known": Prefs().to_dict(),
            "explicit": Prefs().to_dict(),
            "learned": _empty_counts(),
        }
        if not valid_id(user_id):
            return empty
        with self._lock:
            entry = self._read().get(user_id)
            if not entry:
                return empty
            return {
                "known": self._known(entry).to_dict(),
                "explicit": Prefs.from_dict(entry).to_dict(),
                "learned": {**_empty_counts(), **entry.get("counts", {})},
            }

    def delete(self, user_id: str) -> bool:
        if not valid_id(user_id):
            return False
        with self._lock:
            data = self._read()
            existed = data.pop(user_id, None) is not None
            if existed:
                self._write(data)
            return existed
