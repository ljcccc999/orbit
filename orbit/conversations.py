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

    @staticmethod
    def _summary(row: dict[str, Any]) -> str:
        prompts = [
            " ".join(str(message.get("content", "")).split())
            for message in row.get("messages", [])
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        if not prompts:
            return ""
        # Keep the sidebar useful without making another model/API call. The
        # summary combines the user's turns and falls back to the first turn.
        compact = " · ".join(prompts[:4])
        return compact[:96] + ("…" if len(compact) > 96 else "")

    def create(self) -> dict[str, Any]:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        row = {"id": secrets.token_hex(12), "title": "New conversation", "summary": "", "archived": False, "created_at": stamp, "updated_at": stamp, "messages": []}
        self._write(row)
        return row

    def get(self, conversation_id: str) -> dict[str, Any]:
        path = self._path(conversation_id)
        if not path.is_file():
            raise FileNotFoundError("找不到该历史对话")
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, conversation_id: str) -> dict[str, str]:
        path = self._path(conversation_id)
        if not path.is_file():
            raise FileNotFoundError("找不到该历史对话")
        path.unlink()
        return {"status": "deleted", "id": conversation_id}

    def archive(self, conversation_id: str) -> dict[str, str]:
        row = self.get(conversation_id)
        row["archived"] = True
        row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write(row)
        return {"status": "archived", "id": conversation_id}

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("archived") is True:
                    continue
                summary = row.get("summary") or self._summary(row)
                rows.append({
                    "id": row["id"], "title": row.get("title") or "New conversation",
                    "summary": summary, "created_at": row["created_at"], "updated_at": row["updated_at"],
                })
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row["updated_at"], reverse=True)

    def append_exchange(self, conversation_id: str, prompt: str, answer: str) -> dict[str, Any]:
        row = self.get(conversation_id)
        row["messages"].extend(({"role": "user", "content": prompt}, {"role": "assistant", "content": answer}))
        summary = self._summary(row)
        if summary:
            row["summary"] = summary
            first_sentence = " ".join(prompt.split()).split("。", 1)[0].split(".", 1)[0].strip()
            row["title"] = first_sentence[:48] + ("…" if len(first_sentence) > 48 else "")
        row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write(row)
        return row
