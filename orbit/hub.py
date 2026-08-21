from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


class OrbitHubClient:
    """Optional Orbit Hub client. Local-only Orbit remains the default."""

    def __init__(self, data_root: Path):
        self.path = data_root / "hub.json"
        self._lock = threading.RLock()
        self._config = self._load()

    def _load(self) -> dict[str, str]:
        if self.path.is_file():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return {key: str(value.get(key, "")) for key in ("url", "token", "email", "role")}
            except (OSError, json.JSONDecodeError):
                pass
        return {"url": "", "token": "", "email": "", "role": ""}

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    @staticmethod
    def _url(value: str) -> str:
        url = value.strip().rstrip("/")
        parsed = urlparse(url)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("Orbit Hub 必须使用 HTTPS；只有本机测试可以使用 HTTP")
        if not parsed.hostname:
            raise ValueError("Orbit Hub 地址无效")
        return url

    def public_settings(self) -> dict[str, Any]:
        with self._lock:
            return {
                "url": self._config["url"], "email": self._config["email"],
                "role": self._config["role"], "logged_in": bool(self._config["token"]),
            }

    def _request(self, path: str, *, method: str = "GET", payload: Any = None, body: bytes | None = None) -> Any:
        with self._lock:
            base = self._url(self._config["url"])
            token = self._config["token"]
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = body
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read(2000)).get("detail", "")
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = ""
            raise RuntimeError(detail or f"Orbit Hub 返回 HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 Orbit Hub：{exc.reason}") from exc

    def authenticate(self, url: str, email: str, password: str, *, register: bool = False) -> dict[str, Any]:
        with self._lock:
            self._config = {"url": self._url(url), "token": "", "email": "", "role": ""}
        result = self._request("/api/register" if register else "/api/login", method="POST", payload={"email": email, "password": password})
        user = result.get("user") or {}
        with self._lock:
            self._config.update(token=str(result.get("token", "")), email=str(user.get("email", "")), role=str(user.get("role", "")))
            self._save()
        return self.public_settings()

    def logout(self) -> dict[str, Any]:
        try:
            if self._config.get("token"):
                self._request("/api/logout", method="POST", payload={})
        finally:
            with self._lock:
                self._config.update(token="", email="", role="")
                self._save()
        return self.public_settings()

    def upload_model(self, checkpoint: Path, metadata: dict[str, Any], progress: Callable[[int, int, str], None]) -> dict[str, Any]:
        if not self._config.get("token"):
            raise RuntimeError("请先登录 Orbit Hub")
        total = checkpoint.stat().st_size
        digest = hashlib.sha256()
        read = 0
        with checkpoint.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
                read += len(block)
                progress(read, total, "正在计算模型 SHA-256")
        created = self._request("/api/uploads", method="POST", payload={
            "name": metadata.get("name") or checkpoint.stem, "filename": checkpoint.name,
            "size": total, "sha256": digest.hexdigest(), "preset": metadata.get("preset", "custom"),
            "parameters": int(metadata.get("parameters", 0) or 0),
            "description": "Trained locally with Orbit; awaiting administrator review.",
        })
        chunk_size = int(created["chunk_size"])
        sent = 0
        with checkpoint.open("rb") as source:
            for index in range(int(created["chunks"])):
                block = source.read(chunk_size)
                self._request(f"/api/uploads/{created['id']}/chunks/{index}", method="PUT", body=block)
                sent += len(block)
                progress(sent, total, "正在分块上传模型")
        result = self._request(f"/api/uploads/{created['id']}/complete", method="POST", payload={})
        progress(total, total, "上传完成，等待管理员审核")
        return result
