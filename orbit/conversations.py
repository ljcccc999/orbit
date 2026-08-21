from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any


class ConversationStore:
    def __init__(self, data_root: Path):
        self.root = data_root / "conversations"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        if len(conversation_id) != 24 or any(ch not in "0123456789abcdef" for ch in conversation_id):
            raise ValueError("无效的对话编号")
        return self.root / f"{conversation_id}.json"

    def _write(self, row: dict[str, Any]) -> None:
        path = self._path(row["id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def create(self) -> dict[str, Any]:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        row = {"id": secrets.token_hex(12), "title": "New conversation", "created_at": stamp, "updated_at": stamp, "messages": []}
        self._write(row)
        return row

    def get(self, conversation_id: str) -> dict[str, Any]:
        path = self._path(conversation_id)
        if not path.is_file():
            raise FileNotFoundError("找不到该历史对话")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                rows.append({key: row[key] for key in ("id", "title", "created_at", "updated_at")})
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row["updated_at"], reverse=True)

    def append_exchange(self, conversation_id: str, prompt: str, answer: str) -> dict[str, Any]:
        row = self.get(conversation_id)
        row["messages"].extend(({"role": "user", "content": prompt}, {"role": "assistant", "content": answer}))
        if len(row["messages"]) == 2:
            compact = " ".join(prompt.split())
            row["title"] = compact[:48] + ("…" if len(compact) > 48 else "")
        row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write(row)
        return row
