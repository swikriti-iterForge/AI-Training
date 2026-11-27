"""Long-term memory: what a user ASKED us to remember, kept between conversations and restarts.

Deliberately small and transparent: a JSON file you can open, one entry per user id, written only when the
user says "remember ...", readable with "what do you remember", and erased with "forget my preferences" (or
DELETE /memory/{user_id}). It sits behind a Protocol, so swapping it for a database later changes one class.
Not authenticated: any caller who knows a user id can read it. Fine for a local training project, not for production.
"""

import json
import os
import threading
from pathlib import Path
from typing import Protocol

from trailmate_agent.prefs import Prefs


class PreferenceStore(Protocol):
    def load(self, user_id: str) -> Prefs: ...
    def save(self, user_id: str, prefs: Prefs) -> None: ...
    def delete(self, user_id: str) -> bool: ...


class FilePreferenceStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

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

    def load(self, user_id: str) -> Prefs:
        with self._lock:
            return Prefs.from_dict(self._read().get(user_id, {}))

    def save(self, user_id: str, prefs: Prefs) -> None:
        with self._lock:
            data = self._read()
            data[user_id] = prefs.to_dict()
            self._write(data)

    def delete(self, user_id: str) -> bool:
        with self._lock:
            data = self._read()
            existed = data.pop(user_id, None) is not None
            if existed:
                self._write(data)
            return existed
