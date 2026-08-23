from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class OrbitSettings:
    DEFAULTS = {
        "auto_update": False,
        "prevent_sleep": False,
        "background_service": True,
        "computer_control": False,
    }

    def __init__(self, data_root: Path):
        self.path = data_root / "settings.json"
        self._lock = threading.RLock()
        self._values = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return {key: bool(value.get(key, default)) for key, default in self.DEFAULTS.items()}
        except (OSError, json.JSONDecodeError):
            pass
        return dict(self.DEFAULTS)

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
            for key in self.DEFAULTS:
                if key in values:
                    self._values[key] = bool(values[key])
            self._save()
            return dict(self._values)
