from __future__ import annotations

import base64
import difflib
import hashlib
import html
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse


_RUNNING = {"planning", "running", "paused", "waiting_approval"}
_WRITE_TOOLS = {"apply_patch", "shell"}
_SAFE_COMMAND = re.compile(r"^[A-Za-z0-9_./:@%+=,\-\s'\"|&()\[\]{}*?!<>;$`\\]+$")
_BLOCKED_SHELL = re.compile(
    r"(^|\s)(sudo\s+|su\s|rm\s+(?:-[A-Za-z]*r[A-Za-z]*|--recursive)(?=\s|$)|mkfs|diskutil\s+erase|dd\s+if=|shutdown|reboot|halt|poweroff|kill\s+-9\s+1)(\s|$)",
    re.IGNORECASE,
)
_NETWORK_SHELL = re.compile(
    r"(^|[;&|]\s*)(curl|wget|ssh|scp|sftp|git\s+(clone|fetch|pull|push)|pip(?:3)?\s+install|npm\s+(install|publish)|pnpm\s+(install|publish)|yarn\s+(add|install|publish)|brew\s+(install|update|upgrade))\b",
    re.IGNORECASE,
)


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _elapsed(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


class OrbitCodeAgent:
    """A small, inspectable coding-agent loop embedded in the Orbit service.

    Models decide what to do through a strict JSON protocol.  This class owns
    tool execution and approval policy; neither remote nor local models receive
    direct access to the host process.
    """

    def __init__(
        self,
        data_root: Path,
        local_chat: Callable[..., dict[str, Any]],
        list_local_models: Callable[[], list[dict[str, Any]]],
    ) -> None:
        self.root = data_root / "orbit-code"
        self.sessions_root = self.root / "sessions"
        self.attachments_root = self.root / "attachments"
        self.review_root = self.root / "review"
        self.plugins_root = self.root / "plugins"
        self.workspace_root = self.root / "workspace"
        self.settings_path = self.root / "settings.json"
        self.memory_path = self.root / "memory.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.attachments_root.mkdir(parents=True, exist_ok=True)
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.plugins_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._local_chat = local_chat
        self._list_local_models = list_local_models
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._stops: dict[str, threading.Event] = {}
        self._run_gates: dict[str, threading.Event] = {}
        self._approval_events: dict[tuple[str, str], threading.Event] = {}

    def _defaults(self) -> dict[str, Any]:
        return {
            "provider": "local",
            "base_url": "https://api.openai.com/v1",
            "model": "",
            "api_key": "",
            "api_format": "openai",
            "reasoning": "medium",
            "speed": "balanced",
            "permission": "ask",
            "workspace": str(self.workspace_root),
            "workspace_auto": True,
            "capability": "3",
            "active_profile_id": "",
            "profiles": [],
            "local_model_order": [],
            "model_order": [],
            "local_context": True,
            "long_term_memory": True,
            "computer_control": False,
            "enabled_plugins": [],
        }

    def _load_settings(self) -> dict[str, Any]:
        values = self._defaults()
        try:
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                values.update({key: loaded[key] for key in values if key in loaded})
        except (OSError, json.JSONDecodeError):
            pass
        if values.get("provider") == "local" and not str(values.get("model", "")).strip():
            local_models = self._list_local_models()
            if local_models:
                values["model"] = str(local_models[0].get("id", ""))
        return values

    def public_settings(self) -> dict[str, Any]:
        values = self._load_settings()
        key = str(values.pop("api_key", ""))
        profiles = []
        for row in values.get("profiles", []):
            if not isinstance(row, dict):
                continue
            secret = str(row.get("api_key", ""))
            profiles.append({
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "base_url": str(row.get("base_url", "")),
                "model": str(row.get("model", "")),
                "api_format": str(row.get("api_format", "openai")),
                "has_api_key": bool(secret),
                "key_hint": ("••••" + secret[-4:]) if secret else "",
            })
        values["profiles"] = profiles
        local_models = self._list_local_models()
        local_order = [str(item) for item in values.get("local_model_order", [])]
        local_rank = {model_id: index for index, model_id in enumerate(local_order)}
        local_models.sort(key=lambda row: (local_rank.get(str(row.get("id", "")), len(local_rank)), str(row.get("name") or row.get("id") or "").lower()))
        values.update({
            "has_api_key": bool(key),
            "key_hint": ("••••" + key[-4:]) if key else "",
            "local_models": local_models,
        })
        return values

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = self._load_settings()
        for key in ("provider", "base_url", "model", "api_format", "reasoning", "speed", "permission", "workspace", "capability", "active_profile_id"):
            if key in payload:
                values[key] = str(payload[key]).strip()
        for key in ("local_context", "long_term_memory", "computer_control", "workspace_auto"):
            if key in payload:
                values[key] = bool(payload[key])
        if isinstance(payload.get("profile_order"), list):
            requested = [str(item) for item in payload["profile_order"]]
            rows = values.get("profiles", [])
            by_id = {str(row.get("id", "")): row for row in rows}
            values["profiles"] = [by_id[item] for item in requested if item in by_id] + [row for row in rows if str(row.get("id", "")) not in requested]
        if isinstance(payload.get("local_model_order"), list):
            known = {str(row.get("id", "")) for row in self._list_local_models()}
            values["local_model_order"] = [str(item) for item in payload["local_model_order"] if str(item) in known]
        if isinstance(payload.get("model_order"), list):
            api_ids = {"api:" + str(row.get("id", "")) for row in values.get("profiles", [])}
            local_ids = {"local:" + str(row.get("id", "")) for row in self._list_local_models()}
            known = api_ids | local_ids
            requested = [str(item) for item in payload["model_order"] if str(item) in known]
            values["model_order"] = requested + [item for item in values.get("model_order", []) if item in known and item not in requested]
        if str(payload.get("api_key", "")).strip():
            values["api_key"] = str(payload["api_key"]).strip()
        if values["provider"] not in {"local", "api"}:
            raise ValueError("Orbit Code 只支持本地 Orbit、主流 API 或 OpenAI 兼容 API")
        if values["reasoning"] not in {"low", "medium", "high", "xhigh", "max", "ultra"}:
            raise ValueError("不支持的智能等级")
        if values.get("api_format", "openai") not in {"openai", "anthropic"}:
            raise ValueError("不支持的 API 格式")
        if values["speed"] not in {"fast", "balanced", "quality"}:
            raise ValueError("不支持的速度模式")
        if values["permission"] not in {"ask", "workspace", "full"}:
            raise ValueError("不支持的权限模式")
        if str(values["capability"]) not in {"1", "2", "3", "4", "5"}:
            raise ValueError("能力等级必须在 1 到 5 之间")
        workspace = Path(values["workspace"] or self.workspace_root).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError("Orbit Code 工作区不存在")
        values["workspace"] = str(workspace)
        if payload.get("save_profile") is True:
            profile_id = str(payload.get("profile_id", "")).strip()
            rows = values.setdefault("profiles", [])
            selected = next((row for row in rows if row.get("id") == profile_id), None)
            api_key = str(payload.get("api_key", "")).strip()
            profile_values = {
                "name": str(payload.get("profile_name", "")).strip() or str(payload.get("model", "")).strip(),
                "base_url": str(payload.get("base_url", "")).strip(),
                "model": str(payload.get("model", "")).strip(),
                "api_format": str(payload.get("api_format", "openai")).strip() or "openai",
            }
            if profile_values["api_format"] not in {"openai", "anthropic"}:
                raise ValueError("不支持的 API 格式")
            self._api_endpoint(profile_values["base_url"], profile_values["api_format"])
            if not profile_values["model"]:
                raise ValueError("请填写 API 模型名称")
            if selected is None:
                if not api_key:
                    raise ValueError("新 API 配置必须填写 API Key")
                profile_id = secrets.token_hex(8)
                selected = {"id": profile_id, "api_key": api_key}
                rows.append(selected)
            elif api_key:
                selected["api_key"] = api_key
            selected.update(profile_values)
            values["active_profile_id"] = profile_id
            values.update(provider="api", base_url=selected["base_url"], model=selected["model"], api_key=selected["api_key"], api_format=selected["api_format"])
        elif values["provider"] == "api":
            selected = next((row for row in values.get("profiles", []) if row.get("id") == values.get("active_profile_id")), None)
            if selected:
                values.update(base_url=selected["base_url"], model=selected["model"], api_key=selected["api_key"], api_format=selected.get("api_format", "openai"))
            self._api_endpoint(values["base_url"], str(values.get("api_format", "openai")))
            if not values["model"] or not values["api_key"]:
                raise ValueError("请选择已保存的 API 配置")
        temporary = self.settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.settings_path)
        return self.public_settings()

    def set_computer_control(self, enabled: bool) -> dict[str, Any]:
        """Synchronize the host permission without revalidating API profiles."""
        values = self._load_settings()
        values["computer_control"] = bool(enabled)
        temporary = self.settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.settings_path)
        return self.public_settings()

    def delete_profile(self, profile_id: str) -> dict[str, Any]:
        values = self._load_settings()
        rows = [row for row in values.get("profiles", []) if row.get("id") != profile_id]
        if len(rows) == len(values.get("profiles", [])):
            raise FileNotFoundError("找不到 API 配置")
        values["profiles"] = rows
        if values.get("active_profile_id") == profile_id:
            values["active_profile_id"] = rows[0]["id"] if rows else ""
            values["provider"] = "api" if rows else "local"
        temporary = self.settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.settings_path)
        return self.public_settings()

    def list_plugins(self) -> list[dict[str, Any]]:
        enabled = set(self._load_settings().get("enabled_plugins", []))
        rows = []
        for manifest in sorted(self.plugins_root.glob("*/plugin.json")):
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                plugin_id = str(value.get("id") or manifest.parent.name)
                if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", plugin_id):
                    continue
                rows.append({
                    "id": plugin_id,
                    "name": str(value.get("name") or plugin_id)[:120],
                    "version": str(value.get("version") or "0.0.0")[:40],
                    "description": str(value.get("description") or "")[:500],
                    "enabled": plugin_id in enabled,
                })
            except (OSError, json.JSONDecodeError):
                continue
        return rows

    def install_plugin(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        plugin_id = str(manifest.get("id", "")).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", plugin_id):
            raise ValueError("插件 id 只能使用小写字母、数字、点、横线或下划线")
        instructions = str(manifest.get("instructions", "")).strip()
        if not instructions or len(instructions) > 50_000:
            raise ValueError("插件必须包含 instructions，且不能超过 50,000 字")
        clean = {
            "id": plugin_id,
            "name": str(manifest.get("name") or plugin_id)[:120],
            "version": str(manifest.get("version") or "0.0.0")[:40],
            "description": str(manifest.get("description") or "")[:500],
            "instructions": instructions,
        }
        target = self.plugins_root / plugin_id
        target.mkdir(parents=True, exist_ok=True)
        temporary = target / "plugin.tmp"
        temporary.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target / "plugin.json")
        return self.list_plugins()

    def toggle_plugin(self, plugin_id: str, enabled: bool) -> list[dict[str, Any]]:
        if not any(row["id"] == plugin_id for row in self.list_plugins()):
            raise FileNotFoundError("找不到该插件")
        settings = self._load_settings()
        values = set(settings.get("enabled_plugins", []))
        values.add(plugin_id) if enabled else values.discard(plugin_id)
        settings["enabled_plugins"] = sorted(values)
        temporary = self.settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.settings_path)
        return self.list_plugins()

    def _plugin_context(self) -> str:
        enabled = {row["id"] for row in self.list_plugins() if row["enabled"]}
        blocks = []
        for plugin_id in sorted(enabled):
            try:
                value = json.loads((self.plugins_root / plugin_id / "plugin.json").read_text(encoding="utf-8"))
                blocks.append(f"插件 {value.get('name', plugin_id)}：\n{str(value.get('instructions', ''))[:50_000]}")
            except (OSError, json.JSONDecodeError):
                continue
        return "已启用插件：暂无" if not blocks else "已启用插件：\n" + "\n\n".join(blocks)

    @staticmethod
    def _api_endpoint(base_url: str, api_format: str = "openai") -> str:
        value = base_url.strip().rstrip("/")
        parsed = urlparse(value)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("API 必须使用 HTTPS；只有 localhost 可以使用 HTTP")
        if not parsed.hostname:
            raise ValueError("API 地址无效")
        if api_format == "anthropic":
            return value if value.endswith("/messages") else value + "/messages"
        if value.endswith("/chat/completions"):
            return value
        return value + "/chat/completions"

    def _path(self, session_id: str) -> Path:
        if len(session_id) != 24 or any(char not in "0123456789abcdef" for char in session_id):
            raise ValueError("无效的 Orbit Code 会话编号")
        return self.sessions_root / f"{session_id}.json"

    def _save(self, row: dict[str, Any]) -> None:
        path = self._path(row["id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def _event(self, row: dict[str, Any], kind: str, **values: Any) -> dict[str, Any]:
        event = {"id": secrets.token_hex(8), "kind": kind, "time": _stamp(), **values}
        with self._lock:
            row["events"].append(event)
            row["updated_at"] = event["time"]
            self._save(row)
        return event

    def list_sessions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self.sessions_root.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                rows.append({key: row.get(key) for key in ("id", "title", "status", "created_at", "updated_at", "duration_ms")})
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: str(row.get("updated_at", "")), reverse=True)

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if session_id in self._sessions:
                return json.loads(json.dumps(self._sessions[session_id]))
        path = self._path(session_id)
        if not path.is_file():
            raise FileNotFoundError("找不到 Orbit Code 会话")
        return json.loads(path.read_text(encoding="utf-8"))

    def chat_default(self, prompt: str, max_tokens: int = 128, temperature: float = 0.8) -> dict[str, Any]:
        """Answer an Orbit chat with the same default selected by the model library."""
        settings = self._load_settings()
        if settings.get("provider") != "api":
            return self._local_chat(
                prompt,
                model_id=str(settings.get("model") or "") or None,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        if not settings.get("api_key"):
            raise RuntimeError("默认 API 模型尚未保存 API Key")
        system = "You are Orbit, an AI developed by YUNSH. Answer the user's conversation directly and naturally."
        if settings.get("api_format") == "anthropic":
            body = {
                "model": settings["model"],
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            request = urllib.request.Request(
                self._api_endpoint(str(settings["base_url"]), "anthropic"),
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"x-api-key": str(settings["api_key"]), "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
            content = "".join(str(item.get("text", "")) for item in payload.get("content", []) if isinstance(item, dict))
        else:
            body = {
                "model": settings["model"],
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            request = urllib.request.Request(
                self._api_endpoint(str(settings["base_url"]), "openai"),
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {settings['api_key']}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
            choices = payload.get("choices") or []
            content = str((choices[0].get("message") or {}).get("content", "")) if choices else ""
        if not content.strip():
            raise RuntimeError("默认 API 模型没有返回内容")
        return {"model": settings["model"], "model_name": settings["model"], "content": content}

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("请输入要 Orbit Code 完成的任务")
        if len(prompt) > 100_000:
            raise ValueError("单次任务说明不能超过 100,000 字")
        settings = self._load_settings()
        for key in ("provider", "model", "reasoning", "speed", "permission", "workspace"):
            if key in payload and str(payload[key]).strip():
                settings[key] = str(payload[key]).strip()
        for key in ("local_context", "long_term_memory", "computer_control", "workspace_auto"):
            if key in payload:
                settings[key] = bool(payload[key])
        if "capability" in payload:
            settings["capability"] = str(payload["capability"]).strip()
        if settings.get("provider") == "api":
            profile_id = str(payload.get("profile_id") or settings.get("active_profile_id") or "")
            profile = next((item for item in settings.get("profiles", []) if item.get("id") == profile_id), None)
            if profile:
                settings.update(base_url=profile["base_url"], model=profile["model"], api_key=profile["api_key"], api_format=profile.get("api_format", "openai"), active_profile_id=profile_id)
        session_id = secrets.token_hex(12)
        workspace_auto = bool(payload.get("workspace_auto", settings.get("workspace_auto", True)))
        workspace = self.workspace_root if workspace_auto else Path(settings.get("workspace") or self.workspace_root).expanduser().resolve()
        settings["workspace_auto"] = workspace_auto
        if not workspace.is_dir():
            raise ValueError("Orbit Code 工作区不存在")
        settings["workspace"] = str(workspace)
        now = _stamp()
        row: dict[str, Any] = {
            "id": session_id,
            "title": " ".join(prompt.split())[:42],
            "prompt": prompt,
            "status": "planning",
            "created_at": now,
            "updated_at": now,
            "duration_ms": 0,
            "progress": {"completed": 0, "total": 0},
            "changes": {"files": [], "files_changed": 0, "additions": 0, "deletions": 0},
            "settings": {key: settings.get(key) for key in ("provider", "model", "reasoning", "speed", "permission", "workspace", "workspace_auto", "capability", "active_profile_id", "api_format", "local_context", "long_term_memory", "computer_control")},
            "attachments": self._save_attachments(session_id, payload.get("attachments", [])),
            "events": [],
            "history": [],
            "pending_approval": None,
            "directives": [],
        }
        previous_model = str(payload.get("previous_model", "")).strip()
        selected_model = str(settings.get("model") or settings.get("active_profile_id") or "Orbit")
        if previous_model and previous_model != selected_model:
            self._event(
                row, "model_change", title="已更换模型",
                detail=f"{previous_model} → {selected_model}；本轮消息从新模型开始执行。",
                phase="update",
            )
        baseline = None  # computed inside the worker thread so start() returns immediately
        stop = threading.Event()
        run_gate = threading.Event()
        run_gate.set()
        worker = threading.Thread(target=self._run, args=(row, settings, stop, run_gate, baseline), name=f"orbit-code-{session_id[:8]}", daemon=True)
        with self._lock:
            self._sessions[session_id] = row
            self._stops[session_id] = stop
            self._run_gates[session_id] = run_gate
            self._workers[session_id] = worker
            self._save(row)
        worker.start()
        return self.get(session_id)

    def _save_attachments(self, session_id: str, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        target = self.attachments_root / session_id
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        total = 0
        for index, item in enumerate(raw[:8]):
            if not isinstance(item, dict):
                continue
            encoded = str(item.get("data", ""))
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                raise ValueError("图片或语音附件不是有效的 Base64 数据")
            total += len(data)
            if total > 48 * 1024 * 1024:
                raise ValueError("Orbit Code 附件总大小不能超过 48MB")
            name = Path(str(item.get("name") or f"attachment-{index}")).name
            if not name:
                name = f"attachment-{index}"
            path = target / name
            path.write_bytes(data)
            os.chmod(path, 0o600)
            rows.append({"name": name, "type": str(item.get("type", "application/octet-stream")), "path": str(path), "bytes": len(data)})
        return rows

    def approve(self, session_id: str, approved: bool) -> dict[str, Any]:
        restart: tuple[dict[str, Any], dict[str, Any], threading.Event] | None = None
        with self._lock:
            row = self._sessions.get(session_id)
            if row is None:
                row = self.get(session_id)
                self._sessions[session_id] = row
            pending = row.get("pending_approval")
            if not isinstance(pending, dict):
                raise ValueError("当前没有待批准操作")
            pending["decision"] = "approved" if approved else "denied"
            row["status"] = "running" if approved else "stopped"
            self._event(row, "approval_decision", title="已批准" if approved else "已拒绝", detail=pending.get("summary", ""))
            wake = self._approval_events.get((session_id, str(pending.get("id", ""))))
            if wake is not None:
                wake.set()
            elif approved:
                # A service/App update can outlive the persisted approval card
                # while the daemon thread that was waiting for it is gone.  Do
                # not pretend that setting `decision=approved` resumed work:
                # rebuild the worker and grant exactly one matching tool action.
                row["resume_approval"] = {
                    "tool": str(pending.get("tool", "")),
                    "summary": str(pending.get("summary", "")),
                }
                row["pending_approval"] = None
                row["status"] = "planning"
                settings = self._runtime_settings_for_row(row)
                stop = threading.Event()
                run_gate = threading.Event()
                run_gate.set()
                worker = threading.Thread(
                    target=self._run,
                    args=(row, settings, stop, run_gate, None),
                    name=f"orbit-code-{session_id[:8]}-resume",
                    daemon=True,
                )
                self._stops[session_id] = stop
                self._run_gates[session_id] = run_gate
                self._workers[session_id] = worker
                restart = (row, settings, stop)
                self._save(row)
            else:
                row["pending_approval"] = None
                self._event(row, "status", title="已停止", detail="用户拒绝了待批准操作。", phase="summary")
                self._save(row)
        if restart is not None:
            self._workers[session_id].start()
        return self.get(session_id)

    def _runtime_settings_for_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Restore secrets and current provider details for a persisted run."""
        settings = self._load_settings()
        saved = row.get("settings", {})
        if isinstance(saved, dict):
            for key, value in saved.items():
                if value is not None:
                    settings[key] = value
        if settings.get("provider") == "api":
            profile_id = str(settings.get("active_profile_id", ""))
            profile = next(
                (item for item in self._load_settings().get("profiles", []) if item.get("id") == profile_id),
                None,
            )
            if profile:
                settings.update(
                    base_url=profile["base_url"], model=profile["model"], api_key=profile["api_key"],
                    api_format=profile.get("api_format", "openai"), active_profile_id=profile_id,
                )
        return settings

    def stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            stop = self._stops.get(session_id)
            if stop is None:
                raise FileNotFoundError("找不到正在运行的 Orbit Code 会话")
            stop.set()
            row = self._sessions.get(session_id)
            if row is not None and row.get("status") in _RUNNING:
                row["status"] = "stopping"
                self._event(row, "status", title="正在停止", detail="会在当前工具动作的安全边界停止，并保留已经完成的步骤和文件修改。")
        return self.get(session_id)

    def toggle_pause(self, session_id: str) -> dict[str, Any]:
        """Pause or resume at the next safe model/tool boundary."""
        with self._lock:
            row = self._sessions.get(session_id)
            gate = self._run_gates.get(session_id)
            if row is None or gate is None or row.get("status") not in _RUNNING:
                raise FileNotFoundError("找不到可暂停的 Orbit Code 会话")
            if row.get("status") == "paused":
                gate.set()
                row["status"] = "running"
                self._event(row, "status", title="已继续运行", detail="Orbit Code 已从安全边界继续执行。", phase="update")
            else:
                gate.clear()
                row["status"] = "paused"
                self._event(row, "status", title="已暂停", detail="当前动作完成后停在安全边界；再次点击即可继续。", phase="update")
            self._save(row)
        return self.get(session_id)

    @staticmethod
    def _wait_until_resumed(stop: threading.Event, run_gate: threading.Event) -> None:
        while not run_gate.wait(0.1):
            if stop.is_set():
                raise InterruptedError("用户停止了任务")

    def guide(self, session_id: str, prompt: str, mode: str, model_change: dict[str, Any] | None = None) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("引导内容不能为空")
        if mode not in {"steer", "queue"}:
            raise ValueError("引导模式必须是立即引导或排队")
        with self._lock:
            row = self._sessions.get(session_id)
            if row is None:
                row = self.get(session_id)
                self._sessions[session_id] = row
            if row.get("status") not in _RUNNING:
                raise ValueError("该任务已经结束，不能继续引导")
            directive_id = secrets.token_hex(8)
            row.setdefault("directives", []).append({"id": directive_id, "mode": mode, "prompt": prompt, "time": _stamp(), "consumed": False, "deleted": False})
            if isinstance(model_change, dict):
                current = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}
                wanted = {
                    key: model_change.get(key) for key in
                    ("provider", "model", "active_profile_id", "api_format")
                    if model_change.get(key) is not None
                }
                before = str(current.get("model") or current.get("active_profile_id") or "Orbit")
                after = str(wanted.get("model") or wanted.get("active_profile_id") or "Orbit")
                if wanted and (before != after or current.get("provider") != wanted.get("provider")):
                    row["pending_model_change"] = wanted
                    self._event(
                        row, "model_change", title="已更换模型",
                        detail=f"{before} → {after}；从这条引导之后使用新模型。",
                        phase="update",
                    )
            # A live guidance message is itself a turn in the conversation.
            # Put the short acknowledgement in the timeline before the
            # directive marker so the user sees Orbit answer the guidance
            # first, then sees it resume the interrupted plan.
            self._event(
                row,
                "assistant",
                title="先回答引导",
                detail=(
                    f"收到你的引导：“{prompt}”。Orbit 会先回答这一点，"
                    "再从安全边界继续原任务。"
                ),
                phase="guidance_reply",
                directive_id=directive_id,
            )
            self._event(
                row,
                "guidance",
                title="立即引导" if mode == "steer" else "已排队",
                detail=prompt,
                mode=mode,
                directive_id=directive_id,
            )
        return self.get(session_id)

    def update_guidance(self, session_id: str, directive_id: str, action: str, prompt: str = "") -> dict[str, Any]:
        with self._lock:
            row = self._sessions.get(session_id)
            if row is None:
                row = self.get(session_id)
                self._sessions[session_id] = row
            directive = next((item for item in row.get("directives", []) if item.get("id") == directive_id), None)
            if directive is None:
                raise FileNotFoundError("找不到该引导消息")
            if action == "delete":
                if directive.get("mode") == "steer" or directive.get("consumed"):
                    raise ValueError("已经改变位置或执行的引导不可撤销")
                directive["deleted"] = True
            elif action == "queue":
                if directive.get("consumed"):
                    raise ValueError("已经执行的引导不能改为排队")
                directive["mode"] = "queue"
            elif action == "steer":
                if directive.get("consumed"):
                    raise ValueError("已经执行的消息不能改变位置")
                directive["mode"] = "steer"
            elif action == "edit":
                if directive.get("consumed"):
                    raise ValueError("已经执行的引导不能编辑")
                if not prompt.strip():
                    raise ValueError("引导内容不能为空")
                directive["prompt"] = prompt.strip()
                for event in row.get("events", []):
                    if event.get("directive_id") == directive_id:
                        event["detail"] = directive["prompt"]
            else:
                raise ValueError("不支持的引导操作")
            self._save(row)
        return self.get(session_id)

    def _run(self, row: dict[str, Any], settings: dict[str, Any], stop: threading.Event, run_gate: threading.Event, baseline: dict[str, bytes] | None) -> None:
        started = time.monotonic()
        try:
            self._event(row, "status", title="正在理解任务", detail="Orbit Code 正在检查目标、工作区和可用能力。", phase="plan")
            workspace = Path(str(settings["workspace"])).resolve()
            if baseline is None:
                baseline = self._snapshot_workspace(workspace)
            context = self._workspace_context(workspace) if settings.get("local_context", True) else "本地上下文：已关闭"
            memory = self._long_term_context(workspace) if settings.get("long_term_memory", True) else "长期记忆：已关闭"
            plugins = self._plugin_context()
            history: list[dict[str, str]] = []
            attachments = row.get("attachments", [])
            instruction = self._system_prompt(settings, workspace)
            intelligence = self._intelligence_profile(settings)
            user = self._initial_prompt(str(row["prompt"]), context, memory, plugins, attachments, intelligence)
            verification_requested = False
            inspection_requested = False
            tool_counts: dict[str, int] = {}
            for turn in range(intelligence["max_turns"]):
                self._wait_until_resumed(stop, run_gate)
                guidance_turn = bool(row.pop("_guidance_turn", False))
                pending_model = row.pop("pending_model_change", None)
                if isinstance(pending_model, dict):
                    row.setdefault("settings", {}).update(pending_model)
                    settings.update(self._runtime_settings_for_row(row))
                    instruction = self._system_prompt(settings, workspace)
                    with self._lock:
                        self._save(row)
                if stop.is_set():
                    raise InterruptedError("用户停止了任务")
                history_chars = sum(len(item.get("content", "")) for item in history)
                if len(history) > 12 and history_chars > 48_000:
                    compacting = self._event(
                        row, "context_compaction", title="正在自动精简上下文",
                        detail="正在保留关键事实、文件变化、工具结果与未完成事项。",
                        status="running", phase="update",
                    )
                    with self._lock:
                        self._save(row)
                    older, recent = history[:-6], history[-6:]
                    digest_parts = []
                    for item in older:
                        content = " ".join(str(item.get("content", "")).split())
                        if content:
                            digest_parts.append(f"{item.get('role', 'context')}: {content[:900]}")
                    digest = "\n".join(digest_parts)[-12_000:]
                    history = [{"role": "user", "content": "此前执行上下文的自动精简摘要（保留事实、决策、文件、结果和未完成项）：\n" + digest}, *recent]
                    compacting.update(
                        title="已自动精简上下文",
                        detail="Orbit Code 已保留关键事实、文件变化、工具结果和未完成事项，并移除重复的早期记录以降低后续 token 消耗。",
                        status="completed",
                    )
                    with self._lock:
                        self._save(row)
                thinking_started = time.monotonic()
                thinking = self._event(
                    row, "thinking", title="正在思考", detail="正在根据任务、当前观察和用户引导决定下一步。",
                    status="running", phase="thinking",
                )
                try:
                    raw = self._model_interruptible(
                        instruction, user, history, settings, attachments if turn == 0 else [], stop,
                    )
                    with self._lock:
                        # “正在思考”只是一条临时的实时状态，不应在历史时间线里
                        # 变成“已思考”并反复堆积。真正的工具、阶段更新和最终
                        # 回答会继续保留，整轮耗时由完成摘要统一展示。
                        if thinking in row.get("events", []):
                            row["events"].remove(thinking)
                        self._save(row)
                except Exception:
                    with self._lock:
                        if thinking in row.get("events", []):
                            row["events"].remove(thinking)
                        self._save(row)
                    raise
                parse_error: Exception | None = None
                reply: dict[str, Any] | None = None
                for correction_attempt in range(3):
                    try:
                        reply = self._parse_reply(raw)
                        break
                    except RuntimeError as exc:
                        parse_error = exc
                        if correction_attempt >= 2:
                            raise
                        history.append({"role": "assistant", "content": raw})
                        raw = self._model_interruptible(
                            instruction,
                            "上一条回复不符合 Orbit Code 的 JSON 协议。不要解释、不要使用 Markdown；只返回包含 phase、message、actions、done 的一个有效 JSON 对象。",
                            history, settings, [], stop,
                        )
                if reply is None:
                    raise RuntimeError(str(parse_error or "模型没有返回有效协议"))
                message = str(reply.get("message", "")).strip()
                phase = str(reply.get("phase", "update"))
                if message:
                    self._event(
                        row,
                        "assistant",
                        title="引导回答" if guidance_turn else {"plan": "执行计划", "summary": "完成总结"}.get(phase, "阶段更新"),
                        detail=message,
                        phase="guidance_reply" if guidance_turn else phase,
                    )
                actions = reply.get("actions", [])
                if not isinstance(actions, list):
                    actions = []
                history.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
                steer = self._consume_directives(row, "steer")
                if steer:
                    user = (
                        "用户在执行中立即引导：\n"
                        + "\n".join(steer)
                        + "\n请先用自然语言回答这条引导，明确你理解了什么以及它如何影响当前任务；"
                        "回答完成后，再从安全边界继续尚未执行的动作。不要跳过回答直接执行。"
                    )
                    row["_guidance_turn"] = True
                    history.append({"role": "user", "content": user})
                    with self._lock:
                        self._save(row)
                    continue
                if reply.get("done") is True or not actions:
                    if intelligence["rank"] >= 3 and not inspection_requested and sum(tool_counts.get(name, 0) for name in ("list_files", "read_file", "search", "web_search")) == 0:
                        inspection_requested = True
                        user = "在完成前先做一次与任务直接相关的代码/文件定位或资料搜索，不要只凭初始印象总结。"
                        history.append({"role": "user", "content": user})
                        continue
                    if intelligence["rank"] >= 3 and row.get("changes", {}).get("files_changed") and not verification_requested and tool_counts.get("shell", 0) == 0:
                        verification_requested = True
                        user = "已经修改了文件。完成前请运行最相关且安全的检查或测试；若项目确实没有可运行检查，明确说明依据。"
                        history.append({"role": "user", "content": user})
                        continue
                    if phase != "summary":
                        self._event(row, "assistant", title="完成总结", detail=message or "任务已完成。", phase="summary")
                    queued = self._consume_directives(row, "queue")
                    if queued:
                        user = "当前阶段完成后，用户排队了这些后续要求：\n" + "\n".join(queued) + "\n请继续执行并更新计划。"
                        history.append({"role": "user", "content": user})
                        continue
                    row["status"] = "completed"
                    break
                results = []
                row["status"] = "running"
                with self._lock:
                    row["progress"]["total"] = max(
                        int(row["progress"].get("total", 0)),
                        int(row["progress"].get("completed", 0)) + len(actions[:intelligence["max_actions"]]),
                    )
                    self._save(row)
                for action in actions[:intelligence["max_actions"]]:
                    self._wait_until_resumed(stop, run_gate)
                    if stop.is_set():
                        raise InterruptedError("用户停止了任务")
                    if not isinstance(action, dict):
                        continue
                    previous_changes = json.loads(json.dumps(row.get("changes", {})))
                    result = self._execute_with_policy(row, action, settings, workspace, stop)
                    tool_name = str(action.get("tool", ""))
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                    results.append(result)
                    with self._lock:
                        row["progress"]["completed"] = int(row["progress"].get("completed", 0)) + 1
                        row["changes"] = self._workspace_changes(workspace, baseline)
                        self._archive_review_files(row, baseline, workspace)
                        before_files = {item.get("path"): item for item in previous_changes.get("files", [])}
                        changed_files = [
                            item for item in row["changes"].get("files", [])
                            if before_files.get(item.get("path")) != item
                        ]
                        for event in reversed(row.get("events", [])):
                            if event.get("id") == result.get("event_id"):
                                event["file_changes"] = changed_files
                                break
                        self._save(row)
                    steer = self._consume_directives(row, "steer")
                    if steer:
                        results.append({"tool": "guidance", "ok": True, "output": "用户立即引导：" + "\n".join(steer)})
                        break
                user = "工具执行结果：\n" + json.dumps(results, ensure_ascii=False) + "\n根据结果继续；如有新发现，先用阶段更新说明，再执行。"
                history.append({"role": "user", "content": user})
            else:
                raise RuntimeError(f"Orbit Code 达到“{intelligence['label']}”档位的 {intelligence['max_turns']} 轮执行上限；可提高智能档位或缩小任务")
        except InterruptedError as exc:
            row["status"] = "stopped"
            self._event(row, "status", title="已停止", detail=str(exc), phase="summary")
        except Exception as exc:
            row["status"] = "failed"
            error_text = str(exc)
            lowered = error_text.lower()
            quota_markers = (
                "insufficient_quota", "quota", "余额不足", "额度不足", "额度已用完",
                "billing", "payment required", "credit balance", "rate limit exceeded",
            )
            if any(marker in lowered for marker in quota_markers):
                self._event(
                    row, "error", title="API 额度已用完",
                    detail="当前 API Key 没有可用额度，Orbit Code 已安全暂停。请更换 API 配置或前往对应服务商充值后继续。\n\n" + error_text,
                    phase="summary", error_code="api_quota_exhausted",
                )
            else:
                self._event(row, "error", title="执行失败", detail=error_text, phase="summary")
        finally:
            row["duration_ms"] = _elapsed(started)
            row["changes"] = self._workspace_changes(Path(str(settings["workspace"])), baseline)
            self._archive_review_files(row, baseline, Path(str(settings["workspace"])))
            row["updated_at"] = _stamp()
            row["pending_approval"] = None
            with self._lock:
                self._save(row)
            if row.get("status") == "completed" and settings.get("long_term_memory", True):
                self._remember(row)

    def _archive_review_files(self, row: dict[str, Any], baseline: dict[str, bytes], workspace: Path) -> None:
        target = self.review_root / row["id"]
        for change in row.get("changes", {}).get("files", []):
            relative = str(change.get("path", ""))
            if not relative:
                continue
            for version, data in (("before", baseline.get(relative)), ("after", None)):
                if version == "after":
                    try:
                        data = (workspace / relative).read_bytes()
                    except OSError:
                        data = None
                if data is None or len(data) > 2 * 1024 * 1024:
                    continue
                path = target / version / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                os.chmod(path, 0o600)

    def read_session_file(self, session_id: str, relative: str) -> dict[str, Any]:
        row = self.get(session_id)
        workspace = Path(str(row.get("settings", {}).get("workspace", ""))).resolve()
        candidate = (workspace / relative).resolve()
        try:
            normalized = str(candidate.relative_to(workspace))
        except ValueError as exc:
            raise ValueError("文件必须位于该会话工作区内") from exc
        change = next((item for item in row.get("changes", {}).get("files", []) if item.get("path") == normalized), None)
        if change is None:
            raise FileNotFoundError("该文件不在本次修改记录中")
        archived = self.review_root / session_id / ("before" if change.get("status") == "deleted" else "after") / normalized
        source = archived if archived.is_file() else candidate
        if not source.is_file() or source.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("文件不存在、是二进制文件或超过 2MB，不能在面板中完整显示")
        data = source.read_bytes()
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("二进制文件不能在代码面板中显示") from exc
        return {"path": normalized, "status": change.get("status"), "content": content, "diff": change.get("diff", "")}

    def revert_changes(self, session_id: str) -> dict[str, Any]:
        """Restore only files still matching this session's archived after-state."""
        with self._lock:
            row = self.get(session_id)
            if row.get("status") in _RUNNING:
                raise ValueError("任务仍在运行，不能撤销修改")
            if row.get("changes_reverted"):
                raise ValueError("这次修改已经撤销")
            workspace = Path(str(row.get("settings", {}).get("workspace", ""))).resolve()
            changes = list(row.get("changes", {}).get("files", []))
            conflicts: list[str] = []
            for change in changes:
                relative = str(change.get("path", ""))
                candidate = (workspace / relative).resolve()
                try:
                    candidate.relative_to(workspace)
                except ValueError:
                    conflicts.append(relative)
                    continue
                current = candidate.read_bytes() if candidate.is_file() else None
                current_hash = hashlib.sha256(current).hexdigest() if current is not None else None
                if current_hash != change.get("after_sha256"):
                    conflicts.append(relative)
            if conflicts:
                raise ValueError("这些文件在任务结束后又被修改，未撤销：" + "、".join(conflicts[:8]))
            for change in changes:
                relative = str(change["path"])
                candidate = (workspace / relative).resolve()
                before = self.review_root / session_id / "before" / relative
                if change.get("before_sha256") is None:
                    if candidate.exists():
                        candidate.unlink()
                else:
                    if not before.is_file():
                        raise ValueError(f"缺少 {relative} 的修改前副本，不能安全撤销")
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_bytes(before.read_bytes())
            row["changes_reverted"] = True
            row["reverted_at"] = _stamp()
            self._event(row, "revert", title="已撤销文件修改", detail=f"已恢复 {len(changes)} 个文件", phase="summary")
            self._save(row)
            return json.loads(json.dumps(row))

    def _consume_directives(self, row: dict[str, Any], mode: str) -> list[str]:
        with self._lock:
            values = []
            for item in row.get("directives", []):
                if item.get("mode") == mode and item.get("consumed") is not True and item.get("deleted") is not True:
                    item["consumed"] = True
                    values.append(str(item.get("prompt", "")))
            if values:
                self._save(row)
            return values

    @staticmethod
    def _snapshot_workspace(workspace: Path) -> dict[str, bytes]:
        snapshot: dict[str, bytes] = {}
        ignored = {".git", ".venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}
        # Never walk system trees when the workspace is the filesystem root.
        system_roots = {"/System", "/private", "/Library", "/Applications", "/usr", "/opt", "/Volumes", "/cores", "/dev", "/sbin", "/bin", "/etc", "/var", "/tmp", "/Network", "/home", "/nix", "/snap"}
        total = 0
        count = 0
        dirs_seen = 0
        deadline = time.monotonic() + 5.0  # never scan the disk for long
        for root, dirs, files in os.walk(workspace):
            if time.monotonic() > deadline:
                break
            dirs[:] = [name for name in dirs if name not in ignored]
            if str(workspace) == "/":
                dirs[:] = [name for name in dirs if str(Path(workspace) / name) not in system_roots]
            dirs_seen += 1
            if dirs_seen > 4_000:
                break
            for name in files:
                path = Path(root) / name
                try:
                    size = path.stat().st_size
                    if size > 2 * 1024 * 1024 or total + size > 64 * 1024 * 1024:
                        continue
                    data = path.read_bytes()
                except OSError:
                    continue
                snapshot[str(path.relative_to(workspace))] = data
                total += len(data)
                count += 1
                if count >= 20_000:
                    return snapshot
        return snapshot

    @staticmethod
    def _workspace_changes(workspace: Path, baseline: dict[str, bytes]) -> dict[str, Any]:
        current = OrbitCodeAgent._snapshot_workspace(workspace)
        rows: list[dict[str, Any]] = []
        additions = deletions = 0
        for name in sorted(set(baseline) | set(current)):
            before = baseline.get(name)
            after = current.get(name)
            if before == after:
                continue
            status = "added" if before is None else "deleted" if after is None else "modified"
            try:
                old = (before or b"").decode("utf-8").splitlines()
                new = (after or b"").decode("utf-8").splitlines()
                diff_lines = list(difflib.unified_diff(old, new, fromfile=f"a/{name}", tofile=f"b/{name}", lineterm=""))
                added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
                removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
                diff = "\n".join(diff_lines)
            except UnicodeDecodeError:
                added = 1 if after is not None else 0
                removed = 1 if before is not None else 0
                diff = "Binary file changed"
            additions += added
            deletions += removed
            rows.append({
                "path": name,
                "status": status,
                "additions": added,
                "deletions": removed,
                "diff": diff[-120_000:],
                "before_sha256": hashlib.sha256(before).hexdigest() if before is not None else None,
                "after_sha256": hashlib.sha256(after).hexdigest() if after is not None else None,
            })
        return {"files": rows, "files_changed": len(rows), "additions": additions, "deletions": deletions}

    @staticmethod
    def _workspace_context(workspace: Path) -> str:
        names = []
        try:
            names = sorted(entry.name + ("/" if entry.is_dir() else "") for entry in workspace.iterdir())[:120]
        except OSError:
            pass
        references = []
        for name in ("AGENTS.md", "README.md", "README.zh-CN.md"):
            path = workspace / name
            try:
                if path.is_file():
                    references.append(f"{name}：\n{path.read_text(encoding='utf-8', errors='replace')[:8000]}")
            except OSError:
                continue
        suffix = "\n\n" + "\n\n".join(references) if references else ""
        return f"本地上下文：\n工作区：{workspace}\n顶层文件：" + ", ".join(names) + suffix

    def _load_memory(self) -> list[dict[str, Any]]:
        try:
            rows = json.loads(self.memory_path.read_text(encoding="utf-8"))
            return rows if isinstance(rows, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _long_term_context(self, workspace: Path) -> str:
        rows = self._load_memory()
        if not rows:
            return "长期记忆：暂无"
        same = [row for row in rows if row.get("workspace") == str(workspace)]
        other = [row for row in rows if row.get("workspace") != str(workspace)]
        selected = (same[-8:] + other[-4:])[-10:]
        lines = []
        for row in selected:
            files = ", ".join(row.get("files", [])[:12]) or "无文件变化"
            lines.append(f"- {row.get('time', '')}｜{row.get('title', 'Orbit Code')}｜{row.get('summary', '')}｜文件：{files}")
        return "长期记忆（以前完成任务的摘要，可能需要按当前文件重新核验）：\n" + "\n".join(lines)

    def _remember(self, row: dict[str, Any]) -> None:
        if row.get("memory_saved"):
            return
        summaries = [
            str(event.get("detail", "")).strip()
            for event in row.get("events", [])
            if event.get("kind") == "assistant" and event.get("phase") == "summary"
        ]
        summary = summaries[-1] if summaries else "任务已完成。"
        memory = {
            "session_id": row["id"],
            "time": row.get("updated_at", _stamp()),
            "title": str(row.get("title", "Orbit Code"))[:80],
            "summary": summary[:2000],
            "workspace": str(row.get("settings", {}).get("workspace", "")),
            "files": [str(item.get("path", "")) for item in row.get("changes", {}).get("files", []) if item.get("path")][:80],
        }
        with self._lock:
            rows = [item for item in self._load_memory() if item.get("session_id") != row["id"]]
            rows.append(memory)
            temporary = self.memory_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows[-200:], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.memory_path)
            row["memory_saved"] = True
            self._save(row)

    @staticmethod
    def _system_prompt(settings: dict[str, Any], workspace: Path) -> str:
        return """你是 Orbit Code，是由 YUNSH 开发、运行在用户设备上的编码 Agent。你的产品身份始终是 Orbit，不能把自己描述成上游模型、API 提供商或其他产品。
你的职责是把用户提出的问题真正处理完成：在授权范围内理解目标、读取上下文、检查项目、制定方案、搜索代码与网页、修改或创建文件、运行命令、观察结果、定位失败、验证实现、审核差异并清楚汇报。准确性、可复现性、最小意外影响、现有数据安全和用户控制权高于速度。任何文件内容、工具结果、测试通过、发布状态或外部事实都必须来自实际证据；不知道就继续检查或明确不确定，绝不编造。

工作循环：
1. 计划：收到新任务后的第一轮，先用简洁自然的中文告诉用户你理解的目标、准备检查什么、如何实现以及怎样验证；随后才能给出工具动作。计划要具体但不承诺尚未发生的结果。
2. 执行：优先读取最相关的说明、项目状态和目标文件，先定位再修改。保持现有架构和用户未关联的改动，不随意扩大范围，不以重写代替理解。
3. 观察：每次工具执行后都认真读取真实输出，区分成功、失败、部分成功和未知。不要重复已经完成的动作，不要因为命令退出码为零就自动假设产品体验正确。
4. 更新：完成一个有意义的阶段、发现新的根因、改变实现方向、遇到可恢复问题或获得重要证据后，用 update 告诉用户当前发现、已经完成什么和接下来做什么；不要让用户长时间只看到工具日志。
5. 验证：修改后按风险运行相关语法检查、单元测试、构建、静态检查或真实界面验证；界面任务要检查交互与视觉结果，API 任务要检查请求与错误路径，文件任务要审核最终差异。测试受限时明确写出边界。
6. 总结：任务确实完成后才使用 summary 和 done=true。总结必须先给结果，再列出关键改动、验证证据、仍未验证或需要用户完成的事项；不得把计划写成结果，也不得隐藏失败。

工具与操作原则：
- 搜索文件时先用 list_files、search，再按需 read_file；搜索关键词应精确，避免无意义全盘扫描。读取已有项目说明、技能或约束后必须遵守。
- 修改文本优先 apply_patch，使用标准 unified diff；补丁要小而清晰，保留用户已有内容。创建、删除、重命名和大范围替换前核对精确目标。
- shell 用于可复现的检查、构建、测试和项目命令。命令必须明确、范围有限；不得使用危险的宽泛删除、硬重置或覆盖用户工作。
- 外部事实、最新模型、API、设计规范、文献、价格、版本或专业判断需要 web_search 核验；重要结论要多源交叉验证，优先一手资料，同时检查独立实践。搜索结果要查看内容，而不是只看标题。
- 优先使用文件 API、项目 API、命令行和 Shell。只有没有可靠命令行/API 路径、任务确实需要图形界面且用户允许时，才使用 computer 操控鼠标和键盘；每次电脑操作说明目标应用和意图，坐标、按键和输入必须明确，并把成功、失败或权限拒绝记录在执行时间线。
- 用户的批准模式必须生效。需要请求批准时停止在安全边界，不绕过确认；完全访问也不等于允许无关、破坏性或不可恢复操作。撤销或删除排队消息要确认；已提升为即时引导并消费的消息不可撤销。
- 运行中收到引导消息时，先回答这条引导如何影响当前工作，再在安全动作边界调整后续执行；不要中断已经安全运行的独立命令。普通消息默认排队，原任务总结完成后再处理。
- 处理文件修改时，为每个文件记录新增、删除、修改状态与行数；可用时保留 diff，让界面能够以绿色显示新增、红色显示删除。新文件全部算新增，删除文件全部算删除。最终提供可审核的文件清单。
- 每一批工具动作之前都必须先给出可见的 plan 或 update，说明这一批要做什么以及原因；工具完成后，如果得到新发现、完成一个子目标、改变方案或即将进入验证，下一轮先给 update 再调用下一批工具。普通批次控制在 2–4 个紧密相关动作，避免把长任务的全部搜索、编辑和测试塞进同一批。这样界面按“阶段说明 → 可展开的工具汇总 → 下一阶段说明”交错展示，而不是连续堆积日志。
- 阶段更新必须是一段能独立读懂、具有实际信息量的说明，通常用 3–6 句（中文约 90–260 字）交代：刚确认的事实或新发现、证据来自哪里、它对当前任务的影响、这一阶段已经完成什么，以及紧接着要验证或修改什么。不要只写“正在检查”“继续执行”“已修改文件”这类一行状态；也不要把命令清单复述成正文，或泄露隐藏思维链。复杂任务应像成熟工程师的工作记录一样多次出现完整阶段更新，并与每批可展开工具记录交错；简单任务不为凑数量制造空话。
- 阶段说明之后的 actions 只收纳这一段实际对应的读取、搜索、命令或编辑。等这些动作取得结果后，下一轮先写新的完整 update，再给下一批 actions。最终 summary 是唯一真正的最终回答：先给明确结果，再给关键改动、验证证据、文件统计和仍未验证的边界；不要用阶段 update 代替最终总结。
- 设计和界面任务要以清晰层级、可读性、直接操作、键盘可达、可滚动、减少动效和系统一致性为准；不能只改截图效果而破坏导航、焦点、历史、响应式布局或实际点击路径。
- 发现项目已有测试、发布清单、双语文档、版本流程或记忆规则时，将它们视为任务的一部分；但没有用户授权不得擅自发布、删除远程资产、添加协作者或发送外部消息。
- 为提高提示缓存命中率，保持这份稳定身份与规则在前，动态任务、项目上下文、长期记忆、插件和附件只放在用户消息末端；不要在每一轮重复重排稳定规则。节省 token 不能以跳过必要证据和验证为代价。
- 不输出隐藏思维链、内部推理草稿、密钥、完整敏感凭据或不必要的个人数据。只输出用户可验证的计划、阶段发现、动作说明、结果和必要错误。

你的每次回复必须是一个 JSON 对象，不能有 Markdown 围栏：
{{"phase":"plan|update|summary","message":"给用户看的中文说明","actions":[{{"tool":"list_files|read_file|search|web_search|apply_patch|shell|computer","path":"相对路径","query":"搜索词","patch":"unified diff","command":"命令","action":"move|click|double_click|right_click|drag|type|key|hotkey","x":0,"y":0,"to_x":0,"to_y":0,"text":"输入文字","key":"enter","keys":["cmd"],"summary":"动作说明"}}],"done":false}}
JSON 规则：phase 只能是 plan、update 或 summary；message 是直接给用户看的自然语言；actions 是下一批真实动作，每批最多 8 个；工具不需要的字段留空或省略；done 只在不再需要任何工具、验证或必需工作时为 true。需要继续工作但暂时没有动作时 done=false。JSON 之外不得输出任何文字、Markdown 围栏或解释。"""

    @staticmethod
    def _initial_prompt(prompt: str, context: str, memory: str, plugins: str, attachments: list[dict[str, Any]], intelligence: dict[str, Any]) -> str:
        attachment_text = ", ".join(f"{row['name']} ({row['type']}, {row['bytes']} bytes)" for row in attachments) or "无"
        return f"用户任务：{prompt}\n智能档位：{intelligence['label']}。{intelligence['instruction']}\n{context}\n{memory}\n{plugins}\n附件：{attachment_text}\n先说明你要怎么做，然后开始执行。"

    @staticmethod
    def _intelligence_profile(settings: dict[str, Any]) -> dict[str, Any]:
        level = str(settings.get("reasoning", "medium"))
        profiles = {
            # Legacy settings may still contain `none`; treat it as the new Low
            # tier instead of silently increasing its execution budget.
            "none": (0, "低", 5, 3, 700, "只做必要定位、执行和一次关键验证，优先低 token。"),
            "low": (0, "低", 5, 3, 700, "只做必要定位、执行和一次关键验证，优先低 token。"),
            "medium": (1, "中", 9, 5, 1100, "检查相关代码，完成实现并运行必要验证。"),
            "high": (2, "高", 14, 7, 1700, "扩大相关代码搜索，检查相邻影响，并用测试验证修改。"),
            "xhigh": (3, "Pro", 20, 8, 2400, "进行更全面的代码与资料搜索、边界检查和多项验证。"),
            "max": (4, "Max", 28, 8, 3400, "进行充分定位、交叉检查、测试和结果复核；允许更长时间和更多 token。"),
            "ultra": (5, "Ultra", 38, 8, 4800, "使用最高执行预算做深入搜索、实现、测试与复核，适合复杂且高风险的任务。"),
        }
        rank, label, turns, actions, output_tokens, instruction = profiles.get(level, profiles["medium"])
        return {"rank": rank, "label": label, "max_turns": turns, "max_actions": actions, "output_tokens": output_tokens, "instruction": instruction}

    def _model(self, system: str, prompt: str, history: list[dict[str, str]], settings: dict[str, Any], attachments: list[dict[str, Any]]) -> str:
        if settings.get("provider") == "local":
            model = str(settings.get("model") or "").strip() or None
            merged = system + "\n\n" + "\n".join(f"{item['role']}: {item['content']}" for item in history) + "\nuser: " + prompt
            result = self._local_chat(merged, model, self._intelligence_profile(settings)["output_tokens"], 0.2)
            return str(result.get("content", ""))
        return self._api_model(system, prompt, history, settings, attachments)

    def _model_interruptible(
        self,
        system: str,
        prompt: str,
        history: list[dict[str, str]],
        settings: dict[str, Any],
        attachments: list[dict[str, Any]],
        stop: threading.Event,
    ) -> str:
        result: list[str] = []
        error: list[BaseException] = []

        def call() -> None:
            try:
                result.append(self._model(system, prompt, history, settings, attachments))
            except BaseException as exc:  # forwarded to the supervising agent thread
                error.append(exc)

        worker = threading.Thread(target=call, name="orbit-code-model-call", daemon=True)
        worker.start()
        while worker.is_alive():
            if stop.wait(0.1):
                raise InterruptedError("用户停止了当前回答")
            worker.join(0.1)
        if error:
            raise error[0]
        if not result:
            raise RuntimeError("模型调用没有返回结果")
        return result[0]

    def _api_model(self, system: str, prompt: str, history: list[dict[str, str]], settings: dict[str, Any], attachments: list[dict[str, Any]]) -> str:
        if settings.get("api_format") == "anthropic":
            return self._anthropic_model(system, prompt, history, settings, attachments)
        content: Any = prompt
        rich: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in attachments:
            media_type = str(item.get("type", ""))
            data = base64.b64encode(Path(str(item["path"])).read_bytes()).decode()
            if media_type.startswith("image/"):
                rich.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})
            elif media_type.startswith("audio/"):
                fmt = Path(str(item["name"])).suffix.lower().lstrip(".") or "wav"
                rich.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
        if len(rich) > 1:
            content = rich
        body: dict[str, Any] = {
            "model": settings["model"],
            "messages": [{"role": "system", "content": system}, *history, {"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": self._intelligence_profile(settings)["output_tokens"],
            "response_format": {"type": "json_object"},
        }
        effort = settings.get("reasoning", "medium")
        body["reasoning_effort"] = "xhigh" if effort in {"max", "ultra"} else effort
        endpoint = self._api_endpoint(str(settings["base_url"]), "openai")

        def send(payload_body: dict[str, Any]) -> dict[str, Any]:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload_body, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {settings['api_key']}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())

        try:
            payload = send(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            # OpenAI-compatible providers disagree on these optional fields.
            # Retry once with the portable core instead of making every saved
            # provider require a bespoke adapter.
            if exc.code == 400 and ("response_format" in body or "reasoning_effort" in body):
                portable = dict(body)
                portable.pop("response_format", None)
                portable.pop("reasoning_effort", None)
                try:
                    payload = send(portable)
                except urllib.error.HTTPError as retry_exc:
                    retry_detail = retry_exc.read(2000).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Orbit Code API 返回 HTTP {retry_exc.code}：{retry_detail}") from retry_exc
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as retry_exc:
                    raise RuntimeError(f"Orbit Code API 请求失败：{retry_exc}") from retry_exc
            else:
                raise RuntimeError(f"Orbit Code API 返回 HTTP {exc.code}：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Orbit Code API 请求失败：{exc}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Orbit Code API 返回格式不兼容") from exc

    def _anthropic_model(self, system: str, prompt: str, history: list[dict[str, str]], settings: dict[str, Any], attachments: list[dict[str, Any]]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in attachments:
            media_type = str(item.get("type", ""))
            if media_type.startswith("image/"):
                content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(Path(str(item["path"])).read_bytes()).decode()}})
            elif media_type.startswith("audio/"):
                content.append({"type": "text", "text": f"[语音附件 {item.get('name')} 已上传，但该 Anthropic 接口不直接接收音频。]"})
        body = {
            "model": settings["model"],
            "max_tokens": self._intelligence_profile(settings)["output_tokens"],
            "temperature": 0.2,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [*history, {"role": "user", "content": content}],
        }
        request = urllib.request.Request(
            self._api_endpoint(str(settings["base_url"]), "anthropic"),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"x-api-key": str(settings["api_key"]), "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API 返回 HTTP {exc.code}：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Anthropic API 请求失败：{exc}") from exc
        try:
            return "".join(str(item.get("text", "")) for item in payload["content"] if item.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Anthropic API 返回格式不兼容") from exc

    @staticmethod
    def _parse_reply(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            # Some otherwise compatible providers wrap a valid JSON object in
            # a short preamble or a reasoning tag even when JSON mode was
            # requested. Recover the first complete object without accepting
            # arbitrary prose as an executable protocol.
            decoder = json.JSONDecoder()
            value = None
            for index, character in enumerate(text):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(text[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    value = candidate
                    break
            if value is None:
                raise RuntimeError("模型没有返回 Orbit Code 可执行的 JSON 协议；请换更强的模型或提高智能等级") from exc
        if not isinstance(value, dict):
            raise RuntimeError("模型返回的 Orbit Code 协议不是对象")
        return value

    def _execute_with_policy(self, row: dict[str, Any], action: dict[str, Any], settings: dict[str, Any], workspace: Path, stop: threading.Event) -> dict[str, Any]:
        tool = str(action.get("tool", ""))
        if tool not in {"list_files", "read_file", "search", "web_search", "apply_patch", "shell", "computer"}:
            return {"tool": tool, "ok": False, "error": "不支持的工具"}
        summary = str(action.get("summary") or action.get("command") or action.get("path") or tool)
        permission = str(settings.get("permission", "ask"))
        outside = self._action_may_leave_workspace(action, workspace)
        high_risk = self._action_is_high_risk(action)
        # Orbit's first-party web search is available in every mode. This does
        # not grant arbitrary network access to shell subprocesses.
        requires_approval = (
            (permission == "ask" and outside)
            or (permission == "workspace" and high_risk)
        )
        preapproved = False
        resume_approval = row.get("resume_approval")
        if requires_approval and isinstance(resume_approval, dict) and str(resume_approval.get("tool", "")) == tool:
            approved_summary = str(resume_approval.get("summary", "")).strip()
            if not approved_summary or approved_summary in summary or summary in approved_summary:
                preapproved = True
                requires_approval = False
                row["resume_approval"] = None
                with self._lock:
                    self._save(row)
        if requires_approval:
            approval_id = secrets.token_hex(8)
            pending = {"id": approval_id, "summary": summary, "tool": tool, "action": action, "decision": None}
            approval_ready = threading.Event()
            with self._lock:
                row["pending_approval"] = pending
                row["status"] = "waiting_approval"
                self._approval_events[(row["id"], approval_id)] = approval_ready
                self._event(row, "approval", title="请求批准", detail=summary, tool=tool, approval_id=approval_id)
            while not approval_ready.wait(0.15):
                if stop.is_set():
                    with self._lock:
                        self._approval_events.pop((row["id"], approval_id), None)
                    raise InterruptedError("用户停止了任务")
            with self._lock:
                self._approval_events.pop((row["id"], approval_id), None)
                row["pending_approval"] = None
                self._save(row)
            if pending["decision"] != "approved":
                return {"tool": tool, "ok": False, "error": "用户拒绝了操作"}
        event = self._event(
            row, "tool", title=summary, detail="正在执行…", tool=tool, status="running",
            path=str(action.get("path", "")), command=str(action.get("command", "")),
            query=str(action.get("query", "")),
        )
        started = time.monotonic()
        try:
            if tool == "computer" and settings.get("computer_control") is not True:
                raise PermissionError("设置中尚未允许 Orbit Code 操控电脑")
            # Auto-review may approve a routine action outside the project. In
            # Ask mode the same elevation only happens after the user approves.
            elevated = permission == "full" or (outside and permission == "workspace") or requires_approval or preapproved
            output = self._execute(action, workspace, "full" if elevated else permission)
            result = {"tool": tool, "ok": True, "output": output[-20_000:], "duration_ms": _elapsed(started), "event_id": event["id"]}
            event.update(detail=result["output"] or "完成", status="completed", duration_ms=result["duration_ms"])
        except Exception as exc:
            result = {"tool": tool, "ok": False, "error": str(exc), "duration_ms": _elapsed(started), "event_id": event["id"]}
            event.update(detail=str(exc), status="failed", duration_ms=result["duration_ms"])
        with self._lock:
            self._save(row)
        return result

    @staticmethod
    def _action_may_leave_workspace(action: dict[str, Any], workspace: Path) -> bool:
        tool = str(action.get("tool", ""))
        if tool == "web_search":
            return False
        if tool == "computer":
            return True
        if tool == "shell":
            command = str(action.get("command", ""))
            if _NETWORK_SHELL.search(command) or "../" in command:
                return True
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = command.split()
            for index, token in enumerate(tokens):
                value = token.lstrip("<>|")
                if not value.startswith("/") or value == "/dev/null":
                    continue
                # An absolute executable does not by itself grant access to an
                # external file. Arguments and redirection targets still do.
                if index == 0 and value.startswith(("/bin/", "/usr/bin/", "/usr/local/bin/", "/opt/homebrew/bin/")):
                    continue
                path = Path(value).expanduser().resolve()
                if path != workspace and workspace not in path.parents:
                    return True
            executable = tokens[0].lstrip("<>|") if tokens else ""
            for value in re.findall(r"(?<![\w:])/(?!/)[^\s'\";|<>]+", command):
                value = value.rstrip(")],")
                if value in {"/dev/null", executable}:
                    continue
                path = Path(value).expanduser().resolve()
                if path != workspace and workspace not in path.parents:
                    return True
            return False
        if tool == "apply_patch":
            for line in str(action.get("patch", "")).splitlines():
                if not line.startswith(("+++ ", "--- ")):
                    continue
                value = line[4:].split("\t", 1)[0].strip()
                if value == "/dev/null":
                    continue
                if value.startswith(("a/", "b/")):
                    value = value[2:]
                candidate = Path(value or ".").expanduser()
                path = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
                if path != workspace and workspace not in path.parents:
                    return True
            return False
        value = str(action.get("path", "."))
        candidate = Path(value or ".").expanduser()
        path = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
        return path != workspace and workspace not in path.parents

    @staticmethod
    def _action_is_high_risk(action: dict[str, Any]) -> bool:
        tool = str(action.get("tool", ""))
        if tool == "computer":
            return True
        if tool == "shell":
            return bool(_BLOCKED_SHELL.search(str(action.get("command", ""))))
        if tool == "apply_patch":
            patch = str(action.get("patch", ""))
            return "+++ /dev/null" in patch
        return False

    @staticmethod
    def _resolve(workspace: Path, value: str, full: bool) -> Path:
        candidate = Path(value or ".").expanduser()
        path = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
        if not full and path != workspace and workspace not in path.parents:
            raise PermissionError("该权限模式只允许访问当前工作区")
        return path

    def _execute(self, action: dict[str, Any], workspace: Path, permission: str) -> str:
        tool = str(action["tool"])
        full = permission == "full"
        if tool == "list_files":
            path = self._resolve(workspace, str(action.get("path", ".")), full)
            if not path.is_dir():
                raise FileNotFoundError("目录不存在")
            rows = []
            for child in sorted(path.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower()))[:300]:
                rows.append(child.name + ("/" if child.is_dir() else ""))
            return "\n".join(rows)
        if tool == "read_file":
            path = self._resolve(workspace, str(action.get("path", "")), full)
            if not path.is_file():
                raise FileNotFoundError("文件不存在")
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("单次读取文件不能超过 2MB")
            return path.read_text(encoding="utf-8", errors="replace")
        if tool == "search":
            query = str(action.get("query", "")).strip()
            if not query:
                raise ValueError("搜索词不能为空")
            path = self._resolve(workspace, str(action.get("path", ".")), full)
            result = subprocess.run(["rg", "-n", "--hidden", "--glob", "!.git", "--", query, str(path)], capture_output=True, text=True, timeout=30)
            if result.returncode not in {0, 1}:
                raise RuntimeError(result.stderr.strip() or "搜索失败")
            return result.stdout[-20_000:] or "没有匹配结果"
        if tool == "web_search":
            query = str(action.get("query", "")).strip()
            if not query:
                raise ValueError("网页搜索词不能为空")
            request = urllib.request.Request(
                "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
                headers={"User-Agent": "Orbit-Code/0.6 (+https://github.com/ljcccc999/orbit)"},
            )
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    page = response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError(f"网页搜索失败：{exc}") from exc
            rows = self._parse_web_results(page)
            if not rows:
                return "没有找到可读取的网页结果"
            return "\n\n".join(
                f"{index}. {row['title']}\n{row['url']}\n{row['snippet']}"
                for index, row in enumerate(rows[:8], 1)
            )
        if tool == "apply_patch":
            patch = str(action.get("patch", ""))
            if not patch.strip():
                raise ValueError("补丁不能为空")
            if not full and any(line.startswith(("+++ /", "--- /")) for line in patch.splitlines()):
                raise PermissionError("工作区权限不允许补丁使用绝对路径")
            result = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=patch, cwd=workspace, capture_output=True, text=True, timeout=45)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "补丁应用失败")
            return result.stdout.strip() or "补丁已应用"
        if tool == "computer":
            return self._computer_action(action)
        command = str(action.get("command", "")).strip()
        if not command or len(command) > 20_000 or not _SAFE_COMMAND.match(command):
            raise ValueError("Shell 命令为空、过长或含不支持的字符")
        if not full and (_BLOCKED_SHELL.search(command) or "../" in command or "$(`" in command or "$(" in command or "`" in command or re.search(r"(^|\s)/(?!dev/null(?:\s|$))", command)):
            raise PermissionError("该命令需要完全访问模式")
        result = subprocess.run(command, cwd=workspace, shell=True, executable="/bin/zsh", capture_output=True, text=True, timeout=120)
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if result.returncode:
            raise RuntimeError(output or f"命令退出码 {result.returncode}")
        return output or "命令执行完成"

    @staticmethod
    def _computer_action(action: dict[str, Any]) -> str:
        executable = shutil.which("cliclick")
        if not executable:
            raise RuntimeError("当前电脑没有可用的鼠标键盘控制组件")
        name = str(action.get("action", "")).strip()
        x, y = int(action.get("x", 0)), int(action.get("y", 0))
        commands: list[str]
        if name in {"move", "click", "double_click", "right_click"}:
            prefix = {"move": "m", "click": "c", "double_click": "dc", "right_click": "rc"}[name]
            commands = [f"{prefix}:{x},{y}"]
        elif name == "drag":
            commands = [f"dd:{x},{y}", f"dm:{int(action.get('to_x', x))},{int(action.get('to_y', y))}", f"du:{int(action.get('to_x', x))},{int(action.get('to_y', y))}"]
        elif name == "type":
            text = str(action.get("text", ""))
            if not text or len(text) > 10_000:
                raise ValueError("键盘输入内容为空或超过 10,000 字")
            commands = ["t:" + text]
        elif name == "key":
            key = str(action.get("key", "")).lower()
            allowed = {"arrow-down", "arrow-left", "arrow-right", "arrow-up", "delete", "end", "enter", "esc", "home", "page-down", "page-up", "return", "space", "tab"}
            if key not in allowed and not re.fullmatch(r"f(?:[1-9]|1[0-6])", key):
                raise ValueError("不支持的键盘按键")
            commands = ["kp:" + key]
        elif name == "hotkey":
            modifiers = [str(value).lower() for value in action.get("keys", [])]
            if not modifiers or any(value not in {"alt", "cmd", "ctrl", "fn", "shift"} for value in modifiers):
                raise ValueError("快捷键修饰键无效")
            key = str(action.get("key", "")).lower()
            if not re.fullmatch(r"[a-z0-9]", key) and key not in {"return", "enter", "tab", "space", "esc"}:
                raise ValueError("快捷键主键无效")
            commands = ["kd:" + ",".join(modifiers), "t:" + key if re.fullmatch(r"[a-z0-9]", key) else "kp:" + key, "ku:" + ",".join(modifiers)]
        else:
            raise ValueError("不支持的鼠标键盘动作")
        result = subprocess.run([executable, "-w", "35", *commands], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "鼠标键盘动作失败；请检查辅助功能权限")
        return f"鼠标键盘动作已执行：{name}"

    @staticmethod
    def _parse_web_results(page: str) -> list[dict[str, str]]:
        anchors = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippets = re.findall(
            r'<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )
        rows = []
        for index, (raw_url, raw_title) in enumerate(anchors):
            url = html.unescape(raw_url)
            parsed = urlparse(url)
            if "duckduckgo.com" in (parsed.hostname or ""):
                redirect = parse_qs(parsed.query).get("uddg", [])
                if redirect:
                    url = unquote(redirect[0])
            if urlparse(url).scheme not in {"http", "https"}:
                continue
            title = html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
            snippet_raw = snippets[index] if index < len(snippets) else ""
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snippet_raw)).strip()
            rows.append({"title": re.sub(r"\s+", " ", title), "url": url, "snippet": re.sub(r"\s+", " ", snippet)})
        return rows
