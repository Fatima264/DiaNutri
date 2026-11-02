"""Local JSON storage for DiaNutri chat sessions.

Every chat is stored as one JSON file, ``<chats_dir>/<chat_id>.json``. The
default directory is ``saved_chats/`` next to this file. Set the environment
variable ``DIANUTRI_CHATS_DIR`` to store chats somewhere else.

Chat files contain personal health information (diabetic level, allergies,
budget). Keep this directory out of version control and out of any public
deployment. See the Privacy section of the README.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CHATS_DIR = PROJECT_DIRECTORY / "saved_chats"
CHAT_ID_PATTERN = re.compile(r"^chat_[0-9a-f]{10}$")


def chats_dir() -> Path:
    """Return the directory where chats are stored."""
    override = os.environ.get("DIANUTRI_CHATS_DIR")
    return Path(override).expanduser() if override else DEFAULT_CHATS_DIR


def new_chat_id() -> str:
    """Return a new unique chat identifier such as ``chat_1a2b3c4d5e``."""
    return f"chat_{uuid.uuid4().hex[:10]}"


def _chat_path(chat_id: str, directory: Path) -> Path:
    # Validate the id so a crafted value can never point outside the folder.
    if not CHAT_ID_PATTERN.fullmatch(chat_id):
        raise ValueError(f"Invalid chat id: {chat_id!r}")
    return directory / f"{chat_id}.json"


def save_chat(chat_id: str, data: dict[str, Any], directory: Path | None = None) -> None:
    """Write a chat atomically so a crash cannot leave a half-written file."""
    folder = directory or chats_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = _chat_path(chat_id, folder)

    fd, temp_name = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def load_chat(chat_id: str, directory: Path | None = None) -> dict[str, Any] | None:
    """Return a saved chat, or ``None`` if it does not exist or is unreadable."""
    folder = directory or chats_dir()
    try:
        path = _chat_path(chat_id, folder)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def list_chats(directory: Path | None = None) -> list[dict[str, Any]]:
    """Return lightweight metadata for every readable chat, newest first."""
    folder = directory or chats_dir()
    if not folder.exists():
        return []

    chats: list[dict[str, Any]] = []
    for path in folder.glob("chat_*.json"):
        data = load_chat(path.stem, folder)
        if not isinstance(data, dict):
            continue
        chats.append(
            {
                "chat_id": data.get("chat_id", path.stem),
                "created_at": data.get("created_at", ""),
                "answers": data.get("answers", {}) or {},
            }
        )
    # ISO-8601 timestamps sort correctly as plain strings.
    chats.sort(key=lambda chat: chat["created_at"], reverse=True)
    return chats


def delete_chat(chat_id: str, directory: Path | None = None) -> None:
    """Delete a saved chat. Missing chats are ignored."""
    folder = directory or chats_dir()
    try:
        path = _chat_path(chat_id, folder)
    except ValueError:
        return
    path.unlink(missing_ok=True)
