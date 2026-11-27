"""Chat history: every conversation is saved per user, so it can be reopened and continued later.

One JSON file per conversation: data/memory/chats/<user id>/<session id>.json. It holds the messages (with the
trail cards that were shown) and the session's memory (which trails were shown, the preferences so far), so a
reopened chat keeps working: "show me more" still knows what you've already seen, even after a restart.
Only saved when the request carries a user id (the UI's "Remember me" switch). Plain files you can read and delete.
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from trailmate_agent.ids import valid_id

MAX_MESSAGES = 400  # per conversation; older messages are dropped first
MAX_LISTED = 50


class ConversationStore(Protocol):
    def append_turn(
        self,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        trails: list[dict[str, Any]] | None,
        memory: dict[str, Any] | None,
    ) -> None: ...
    def list_chats(self, user_id: str) -> list[dict[str, Any]]: ...
    def get(self, user_id: str, session_id: str) -> dict[str, Any] | None: ...
    def delete(self, user_id: str, session_id: str) -> bool: ...
    def delete_all(self, user_id: str) -> int: ...


def _now() -> str:
    return datetime.now(UTC).isoformat(
        timespec="microseconds"
    )  # fine enough that two chats never tie


class FileConversationStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()

    def _folder(self, user_id: str) -> Path | None:
        if not valid_id(user_id):
            return None
        folder = (self._root / user_id).resolve()
        return (
            folder if folder.is_relative_to(self._root.resolve()) else None
        )  # never leave the data folder

    def _file(self, user_id: str, session_id: str) -> Path | None:
        folder = self._folder(user_id)
        return folder / f"{session_id}.json" if folder and valid_id(session_id) else None

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def append_turn(self, user_id, session_id, user_text, assistant_text, trails, memory) -> None:
        path = self._file(user_id, session_id)
        if path is None:
            return
        with self._lock:
            chat = self._read(path) or {
                "id": session_id,
                "user": user_id,
                "title": " ".join(user_text.split())[:48] or "New chat",
                "created": _now(),
                "messages": [],
            }
            stamp = _now()
            chat["messages"] += [
                {"role": "user", "text": user_text, "ts": stamp},
                {
                    "role": "assistant",
                    "text": assistant_text,
                    "trails": trails or None,
                    "ts": stamp,
                },
            ]
            chat["messages"] = chat["messages"][-MAX_MESSAGES:]
            chat["updated"] = stamp
            if memory is not None:
                chat["memory"] = memory
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(chat, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)  # never leaves a half-written file

    def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        folder = self._folder(user_id)
        if folder is None or not folder.is_dir():
            return []
        with self._lock:
            chats = [c for p in folder.glob("*.json") if (c := self._read(p))]
        chats.sort(key=lambda c: c.get("updated", ""), reverse=True)
        return [
            {
                "id": c["id"],
                "title": c.get("title", "Chat"),
                "updated": c.get("updated", ""),
                "messages": len(c.get("messages", [])),
            }
            for c in chats[:MAX_LISTED]
        ]

    def get(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        path = self._file(user_id, session_id)
        with self._lock:
            return self._read(path) if path and path.exists() else None

    def delete(self, user_id: str, session_id: str) -> bool:
        path = self._file(user_id, session_id)
        with self._lock:
            if path is None or not path.exists():
                return False
            path.unlink()
            return True

    def delete_all(self, user_id: str) -> int:
        folder = self._folder(user_id)
        if folder is None or not folder.is_dir():
            return 0
        with self._lock:
            files = list(folder.glob("*.json"))
            for p in files:
                p.unlink()
            return len(files)
