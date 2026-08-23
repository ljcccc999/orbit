from __future__ import annotations

import base64
import difflib
import hashlib
import html
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse


_RUNNING = {"planning", "running", "waiting_approval"}
_WRITE_TOOLS = {"apply_patch", "shell"}
_SAFE_COMMAND = re.compile(r"^[A-Za-z0-9_./:@%+=,\-\s'\"|&()\[\]{}*?!<>;$`\\]+$")
_BLOCKED_SHELL = re.compile(
    r"(^|\s)(sudo\s+|su\s|rm\s+-[^\n]*r|mkfs|diskutil\s+erase|dd\s+if=|shutdown|reboot|halt|poweroff|kill\s+-9\s+1)(\s|$)",
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
        self.settings_path = self.root / "settings.json"
        self.memory_path = self.root / "memory.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.attachments_root.mkdir(parents=True, exist_ok=True)
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.plugins_root.mkdir(parents=True, exist_ok=True)
        self._local_chat = local_chat
        self._list_local_models = list_local_models
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._stops: dict[str, threading.Event] = {}

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "provider": "local",
            "base_url": "https://api.openai.com/v1",
            "model": "",
            "api_key": "",
            "api_format": "openai",
            "reasoning": "medium",
            "speed": "balanced",
            "permission": "ask",
            "workspace": "",
            "capability": "3",
            "active_profile_id": "",
            "profiles": [],
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
        values.update({
            "has_api_key": bool(key),
            "key_hint": ("••••" + key[-4:]) if key else "",
            "local_models": self._list_local_models(),
        })
        return values

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = self._load_settings()
        for key in ("provider", "base_url", "model", "api_format", "reasoning", "speed", "permission", "workspace", "capability", "active_profile_id"):
            if key in payload:
                values[key] = str(payload[key]).strip()
        for key in ("local_context", "long_term_memory", "computer_control"):
            if key in payload:
                values[key] = bool(payload[key])
        if str(payload.get("api_key", "")).strip():
            values["api_key"] = str(payload["api_key"]).strip()
        if values["provider"] not in {"local", "api"}:
            raise ValueError("Orbit Code 只支持本地 Orbit、主流 API 或 OpenAI 兼容 API")
        if values["reasoning"] not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("不支持的智能等级")
        if values.get("api_format", "openai") not in {"openai", "anthropic"}:
            raise ValueError("不支持的 API 格式")
        if values["speed"] not in {"fast", "balanced", "quality"}:
            raise ValueError("不支持的速度模式")
        if values["permission"] not in {"ask", "workspace", "full"}:
            raise ValueError("不支持的权限模式")
        if str(values["capability"]) not in {"1", "2", "3", "4", "5"}:
            raise ValueError("能力等级必须在 1 到 5 之间")
        workspace = Path(values["workspace"] or os.getcwd()).expanduser().resolve()
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
        for key in ("local_context", "long_term_memory", "computer_control"):
            if key in payload:
                settings[key] = bool(payload[key])
        if "capability" in payload:
            settings["capability"] = str(payload["capability"]).strip()
        if settings.get("provider") == "api":
            profile_id = str(payload.get("profile_id") or settings.get("active_profile_id") or "")
            profile = next((item for item in settings.get("profiles", []) if item.get("id") == profile_id), None)
            if profile:
                settings.update(base_url=profile["base_url"], model=profile["model"], api_key=profile["api_key"], api_format=profile.get("api_format", "openai"), active_profile_id=profile_id)
        workspace = Path(settings.get("workspace") or os.getcwd()).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError("Orbit Code 工作区不存在")
        settings["workspace"] = str(workspace)
        session_id = secrets.token_hex(12)
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
            "settings": {key: settings.get(key) for key in ("provider", "model", "reasoning", "speed", "permission", "workspace", "capability", "active_profile_id", "api_format", "local_context", "long_term_memory", "computer_control")},
            "attachments": self._save_attachments(session_id, payload.get("attachments", [])),
            "events": [],
            "history": [],
            "pending_approval": None,
            "directives": [],
        }
        baseline = self._snapshot_workspace(workspace)
        stop = threading.Event()
        worker = threading.Thread(target=self._run, args=(row, settings, stop, baseline), name=f"orbit-code-{session_id[:8]}", daemon=True)
        with self._lock:
            self._sessions[session_id] = row
            self._stops[session_id] = stop
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
        with self._lock:
            row = self._sessions.get(session_id)
            if row is None:
                row = self.get(session_id)
                self._sessions[session_id] = row
            pending = row.get("pending_approval")
            if not isinstance(pending, dict):
                raise ValueError("当前没有待批准操作")
            pending["decision"] = "approved" if approved else "denied"
            row["status"] = "running"
            self._event(row, "approval_decision", title="已批准" if approved else "已拒绝", detail=pending.get("summary", ""))
        return self.get(session_id)

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

    def guide(self, session_id: str, prompt: str, mode: str) -> dict[str, Any]:
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

    def _run(self, row: dict[str, Any], settings: dict[str, Any], stop: threading.Event, baseline: dict[str, bytes]) -> None:
        started = time.monotonic()
        try:
            self._event(row, "status", title="正在理解任务", detail="Orbit Code 正在检查目标、工作区和可用能力。", phase="plan")
            workspace = Path(str(settings["workspace"])).resolve()
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
                if stop.is_set():
                    raise InterruptedError("用户停止了任务")
                thinking_started = time.monotonic()
                thinking = self._event(
                    row, "thinking", title="正在思考", detail="正在根据任务、当前观察和用户引导决定下一步。",
                    status="running", phase="thinking",
                )
                try:
                    raw = self._model_interruptible(
                        instruction, user, history, settings, attachments if turn == 0 else [], stop,
                    )
                    thinking.update(title="已思考", status="completed", duration_ms=_elapsed(thinking_started))
                    with self._lock:
                        self._save(row)
                except Exception:
                    thinking.update(title="思考已停止" if stop.is_set() else "思考失败", status="stopped" if stop.is_set() else "failed", duration_ms=_elapsed(thinking_started))
                    with self._lock:
                        self._save(row)
                    raise
                reply = self._parse_reply(raw)
                message = str(reply.get("message", "")).strip()
                phase = str(reply.get("phase", "update"))
                if message:
                    self._event(row, "assistant", title={"plan": "执行计划", "summary": "完成总结"}.get(phase, "阶段更新"), detail=message, phase=phase)
                actions = reply.get("actions", [])
                if not isinstance(actions, list):
                    actions = []
                history.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
                steer = self._consume_directives(row, "steer")
                if steer:
                    user = "用户在执行中立即引导：\n" + "\n".join(steer) + "\n请据此重新规划，尚未执行的动作不要继续。"
                    history.append({"role": "user", "content": user})
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
            self._event(row, "error", title="执行失败", detail=str(exc), phase="summary")
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
        total = 0
        for root, dirs, files in os.walk(workspace):
            dirs[:] = [name for name in dirs if name not in ignored]
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
        return """你是 Orbit Code，由 YUNSH 开发的编码 Agent。
必须按计划、执行、观察、更新、验证、总结循环完成任务。第一轮先用简洁自然语言告诉用户你准备怎么做，然后给出工具动作。执行一部分或发现新信息后，用 update 阶段说明。完成时用 summary 总结实际结果和验证，不得虚构成功。
你的每次回复必须是一个 JSON 对象，不能有 Markdown 围栏：
{{"phase":"plan|update|summary","message":"给用户看的中文说明","actions":[{{"tool":"list_files|read_file|search|web_search|apply_patch|shell|computer","path":"相对路径","query":"搜索词","patch":"unified diff","command":"命令","action":"move|click|double_click|right_click|drag|type|key|hotkey","x":0,"y":0,"to_x":0,"to_y":0,"text":"输入文字","key":"enter","keys":["cmd"],"summary":"动作说明"}}],"done":false}}
规则：优先 list_files/search/read_file 后再改；apply_patch 使用标准 unified diff；优先使用可复现的文件 API、命令行和 Shell，只有没有可靠命令行/API 路径且完成任务确实需要图形界面时才使用 computer；computer 坐标和输入必须明确；每次 actions 最多 8 个；没有工具要运行或任务完成时 done=true。不要输出隐藏思维链，只给计划、发现、动作说明和结果。"""

    @staticmethod
    def _initial_prompt(prompt: str, context: str, memory: str, plugins: str, attachments: list[dict[str, Any]], intelligence: dict[str, Any]) -> str:
        attachment_text = ", ".join(f"{row['name']} ({row['type']}, {row['bytes']} bytes)" for row in attachments) or "无"
        return f"用户任务：{prompt}\n智能档位：{intelligence['label']}。{intelligence['instruction']}\n{context}\n{memory}\n{plugins}\n附件：{attachment_text}\n先说明你要怎么做，然后开始执行。"

    @staticmethod
    def _intelligence_profile(settings: dict[str, Any]) -> dict[str, Any]:
        level = str(settings.get("reasoning", "medium"))
        profiles = {
            "none": (0, "即时", 4, 3, 640, "只做必要定位和一次关键验证，优先速度与低 token。"),
            "low": (1, "轻量", 7, 4, 900, "做针对性搜索、实现和关键验证，控制执行范围。"),
            "medium": (2, "标准", 12, 6, 1400, "检查相关代码，完成实现并运行必要测试。"),
            "high": (3, "深入", 18, 8, 2100, "扩大相关代码搜索，检查相邻影响，并用测试验证修改。"),
            "xhigh": (4, "全面", 24, 8, 3000, "进行更全面的代码与资料搜索、边界检查和多项验证；允许更长时间和更多 token。"),
            "max": (5, "最大", 32, 8, 4000, "在不重复工作的前提下进行最充分的定位、交叉检查、测试和结果复核；耗时与 token 最高。"),
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
        if settings.get("reasoning") != "none":
            body["reasoning_effort"] = "xhigh" if settings.get("reasoning") == "max" else settings.get("reasoning", "medium")
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
        requires_approval = (tool in _WRITE_TOOLS and permission == "ask") or (tool == "computer" and permission != "full")
        if requires_approval:
            approval_id = secrets.token_hex(8)
            pending = {"id": approval_id, "summary": summary, "tool": tool, "decision": None}
            with self._lock:
                row["pending_approval"] = pending
                row["status"] = "waiting_approval"
                self._event(row, "approval", title="请求批准", detail=summary, tool=tool, approval_id=approval_id)
            while pending["decision"] is None:
                if stop.wait(0.15):
                    raise InterruptedError("用户停止了任务")
            row["pending_approval"] = None
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
            output = self._execute(action, workspace, permission)
            result = {"tool": tool, "ok": True, "output": output[-20_000:], "duration_ms": _elapsed(started), "event_id": event["id"]}
            event.update(detail=result["output"] or "完成", status="completed", duration_ms=result["duration_ms"])
        except Exception as exc:
            result = {"tool": tool, "ok": False, "error": str(exc), "duration_ms": _elapsed(started), "event_id": event["id"]}
            event.update(detail=str(exc), status="failed", duration_ms=result["duration_ms"])
        with self._lock:
            self._save(row)
        return result

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
