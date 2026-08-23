from __future__ import annotations

import base64
import difflib
import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


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
        self.settings_path = self.root / "settings.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.attachments_root.mkdir(parents=True, exist_ok=True)
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
            "reasoning": "medium",
            "speed": "balanced",
            "permission": "ask",
            "workspace": "",
            "capability": "3",
            "active_profile_id": "",
            "profiles": [],
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
        for key in ("provider", "base_url", "model", "reasoning", "speed", "permission", "workspace", "capability", "active_profile_id"):
            if key in payload:
                values[key] = str(payload[key]).strip()
        if str(payload.get("api_key", "")).strip():
            values["api_key"] = str(payload["api_key"]).strip()
        if values["provider"] not in {"local", "api"}:
            raise ValueError("Orbit Code 只支持本地 Orbit 或 OpenAI 兼容 API")
        if values["reasoning"] not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("不支持的思考深度")
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
            }
            self._api_endpoint(profile_values["base_url"])
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
            values.update(provider="api", base_url=selected["base_url"], model=selected["model"], api_key=selected["api_key"])
        elif values["provider"] == "api":
            selected = next((row for row in values.get("profiles", []) if row.get("id") == values.get("active_profile_id")), None)
            if selected:
                values.update(base_url=selected["base_url"], model=selected["model"], api_key=selected["api_key"])
            self._api_endpoint(values["base_url"])
            if not values["model"] or not values["api_key"]:
                raise ValueError("请选择已保存的 API 配置")
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

    @staticmethod
    def _api_endpoint(base_url: str) -> str:
        value = base_url.strip().rstrip("/")
        parsed = urlparse(value)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("API 必须使用 HTTPS；只有 localhost 可以使用 HTTP")
        if not parsed.hostname:
            raise ValueError("API 地址无效")
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
        if "capability" in payload:
            settings["capability"] = str(payload["capability"]).strip()
        if settings.get("provider") == "api":
            profile_id = str(payload.get("profile_id") or settings.get("active_profile_id") or "")
            profile = next((item for item in settings.get("profiles", []) if item.get("id") == profile_id), None)
            if profile:
                settings.update(base_url=profile["base_url"], model=profile["model"], api_key=profile["api_key"], active_profile_id=profile_id)
        workspace = Path(settings.get("workspace") or os.getcwd()).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError("Orbit Code 工作区不存在")
        settings["workspace"] = str(workspace)
        session_id = secrets.token_hex(12)
        now = _stamp()
        row: dict[str, Any] = {
            "id": session_id,
            "title": " ".join(prompt.split())[:72],
            "prompt": prompt,
            "status": "planning",
            "created_at": now,
            "updated_at": now,
            "duration_ms": 0,
            "progress": {"completed": 0, "total": 0},
            "changes": {"files": [], "files_changed": 0, "additions": 0, "deletions": 0},
            "settings": {key: settings.get(key) for key in ("provider", "model", "reasoning", "speed", "permission", "workspace", "capability", "active_profile_id")},
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
                directive["deleted"] = True
            elif action == "queue":
                if directive.get("consumed"):
                    raise ValueError("已经执行的引导不能改为排队")
                directive["mode"] = "queue"
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
            context = self._workspace_context(workspace)
            history: list[dict[str, str]] = []
            attachments = row.get("attachments", [])
            instruction = self._system_prompt(settings, workspace)
            user = self._initial_prompt(str(row["prompt"]), context, attachments)
            for turn in range(16):
                if stop.is_set():
                    raise InterruptedError("用户停止了任务")
                raw = self._model(instruction, user, history, settings, attachments if turn == 0 else [])
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
                    queued = self._consume_directives(row, "queue")
                    if queued:
                        user = "当前阶段完成后，用户排队了这些后续要求：\n" + "\n".join(queued) + "\n请继续执行并更新计划。"
                        history.append({"role": "user", "content": user})
                        continue
                    if phase != "summary":
                        self._event(row, "assistant", title="完成总结", detail=message or "任务已完成。", phase="summary")
                    row["status"] = "completed"
                    break
                results = []
                row["status"] = "running"
                with self._lock:
                    row["progress"]["total"] = max(
                        int(row["progress"].get("total", 0)),
                        int(row["progress"].get("completed", 0)) + len(actions[:8]),
                    )
                    self._save(row)
                for action in actions[:8]:
                    if stop.is_set():
                        raise InterruptedError("用户停止了任务")
                    if not isinstance(action, dict):
                        continue
                    result = self._execute_with_policy(row, action, settings, workspace, stop)
                    results.append(result)
                    with self._lock:
                        row["progress"]["completed"] = int(row["progress"].get("completed", 0)) + 1
                        row["changes"] = self._workspace_changes(workspace, baseline)
                        self._save(row)
                    steer = self._consume_directives(row, "steer")
                    if steer:
                        results.append({"tool": "guidance", "ok": True, "output": "用户立即引导：" + "\n".join(steer)})
                        break
                queued = self._consume_directives(row, "queue")
                if queued:
                    results.append({"tool": "queued_guidance", "ok": True, "output": "\n".join(queued)})
                user = "工具执行结果：\n" + json.dumps(results, ensure_ascii=False) + "\n根据结果继续；如有新发现，先用阶段更新说明，再执行。"
                history.append({"role": "user", "content": user})
            else:
                raise RuntimeError("Orbit Code 达到 16 轮执行上限；请缩小任务或继续新会话")
        except InterruptedError as exc:
            row["status"] = "stopped"
            self._event(row, "status", title="已停止", detail=str(exc), phase="summary")
        except Exception as exc:
            row["status"] = "failed"
            self._event(row, "error", title="执行失败", detail=str(exc), phase="summary")
        finally:
            row["duration_ms"] = _elapsed(started)
            row["changes"] = self._workspace_changes(Path(str(settings["workspace"])), baseline)
            row["updated_at"] = _stamp()
            row["pending_approval"] = None
            with self._lock:
                self._save(row)

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
        return f"工作区：{workspace}\n顶层文件：" + ", ".join(names)

    @staticmethod
    def _system_prompt(settings: dict[str, Any], workspace: Path) -> str:
        speed = {"fast": "快速：优先少量关键检查", "balanced": "平衡：实现并做必要验证", "quality": "质量优先：充分检查和验证"}.get(str(settings.get("speed")), "平衡")
        return f"""你是 Orbit Code，由 YUNSH 开发的编码 Agent。你在 {workspace} 工作。
必须按计划、执行、观察、更新、验证、总结循环完成任务。第一轮先用简洁自然语言告诉用户你准备怎么做，然后给出工具动作。执行一部分或发现新信息后，用 update 阶段说明。完成时用 summary 总结实际结果和验证，不得虚构成功。
模式：{speed}。思考深度：{settings.get('reasoning', 'medium')}。能力等级：{settings.get('capability', '3')}/5；等级越高越允许拆分更多必要步骤，但仍须停止重复工作。
你的每次回复必须是一个 JSON 对象，不能有 Markdown 围栏：
{{"phase":"plan|update|summary","message":"给用户看的中文说明","actions":[{{"tool":"list_files|read_file|search|apply_patch|shell","path":"相对路径","query":"搜索词","patch":"unified diff","command":"命令","summary":"动作说明"}}],"done":false}}
规则：优先 list_files/search/read_file 后再改；apply_patch 使用标准 unified diff；shell 仅用于必要命令和测试；每次 actions 最多 8 个；没有工具要运行或任务完成时 done=true。不要输出隐藏思维链，只给计划、发现、动作说明和结果。"""

    @staticmethod
    def _initial_prompt(prompt: str, context: str, attachments: list[dict[str, Any]]) -> str:
        attachment_text = ", ".join(f"{row['name']} ({row['type']}, {row['bytes']} bytes)" for row in attachments) or "无"
        return f"用户任务：{prompt}\n{context}\n附件：{attachment_text}\n先说明你要怎么做，然后开始执行。"

    def _model(self, system: str, prompt: str, history: list[dict[str, str]], settings: dict[str, Any], attachments: list[dict[str, Any]]) -> str:
        if settings.get("provider") == "local":
            model = str(settings.get("model") or "").strip() or None
            merged = system + "\n\n" + "\n".join(f"{item['role']}: {item['content']}" for item in history[-12:]) + "\nuser: " + prompt
            result = self._local_chat(merged, model, 384, 0.2)
            return str(result.get("content", ""))
        return self._api_model(system, prompt, history, settings, attachments)

    def _api_model(self, system: str, prompt: str, history: list[dict[str, str]], settings: dict[str, Any], attachments: list[dict[str, Any]]) -> str:
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
            "messages": [{"role": "system", "content": system}, *history[-12:], {"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": {"fast": 900, "balanced": 1600, "quality": 2600}.get(str(settings.get("speed")), 1600),
            "response_format": {"type": "json_object"},
        }
        if settings.get("reasoning") != "none":
            body["reasoning_effort"] = settings.get("reasoning", "medium")
        request = urllib.request.Request(
            self._api_endpoint(str(settings["base_url"])),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {settings['api_key']}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise RuntimeError(f"Orbit Code API 返回 HTTP {exc.code}：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Orbit Code API 请求失败：{exc}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Orbit Code API 返回格式不兼容") from exc

    @staticmethod
    def _parse_reply(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("模型没有返回 Orbit Code 可执行的 JSON 协议；请换更强的模型或提高思考深度") from exc
        if not isinstance(value, dict):
            raise RuntimeError("模型返回的 Orbit Code 协议不是对象")
        return value

    def _execute_with_policy(self, row: dict[str, Any], action: dict[str, Any], settings: dict[str, Any], workspace: Path, stop: threading.Event) -> dict[str, Any]:
        tool = str(action.get("tool", ""))
        if tool not in {"list_files", "read_file", "search", "apply_patch", "shell"}:
            return {"tool": tool, "ok": False, "error": "不支持的工具"}
        summary = str(action.get("summary") or action.get("command") or action.get("path") or tool)
        permission = str(settings.get("permission", "ask"))
        requires_approval = tool in _WRITE_TOOLS and permission == "ask"
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
        event = self._event(row, "tool", title=summary, detail="正在执行…", tool=tool, status="running")
        started = time.monotonic()
        try:
            output = self._execute(action, workspace, permission)
            result = {"tool": tool, "ok": True, "output": output[-20_000:], "duration_ms": _elapsed(started)}
            event.update(detail=result["output"] or "完成", status="completed", duration_ms=result["duration_ms"])
        except Exception as exc:
            result = {"tool": tool, "ok": False, "error": str(exc), "duration_ms": _elapsed(started)}
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
