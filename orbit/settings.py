from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class OrbitSettings:
    def __init__(self, data_root: Path):
        self.path = data_root / "settings.json"
        self._lock = threading.RLock()
        self._values = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return {"auto_update": bool(value.get("auto_update", False))}
        except (OSError, json.JSONDecodeError):
            pass
        return {"auto_update": False}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if "auto_update" in values:
                self._values["auto_update"] = bool(values["auto_update"])
            self._save()
            return dict(self._values)

