from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import OrbitConfig
from .jobs import create_job_bundle
from .runtime import OrbitRuntime
from .training_config import TrainingConfig
from .web_ui import PAGE


MAX_REQUEST_BYTES = 64 * 1024 * 1024


class OrbitHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: OrbitRuntime):
        super().__init__(address, Handler)
        self.runtime = runtime
        self.verbose = False

    def server_bind(self) -> None:
        # HTTPServer normally performs reverse DNS after binding. A local-only
        # API must not depend on DNS availability, and this can otherwise delay
        # startup for tens of seconds on an offline or proxied computer.
        socketserver.TCPServer.server_bind(self)
        self.server_name = str(self.server_address[0])
        self.server_port = int(self.server_address[1])


class Handler(BaseHTTPRequestHandler):
    server: OrbitHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin", "")
        if origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:"):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self._headers(status, content_type, len(body))
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, exc: Exception | str) -> None:
        self._json(status, {"error": str(exc)})

    def _read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 0 or size > MAX_REQUEST_BYTES:
            raise ValueError("请求内容过大，单次最多 64MB")
        if not size:
            return {}
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("请求必须是 JSON 对象")
        return value

    def _require_api_key(self, model: str | None = None) -> dict[str, str] | None:
        supplied = self.headers.get("Authorization", "")
        if supplied.lower().startswith("bearer "):
            supplied = supplied[7:].strip()
        record = self.server.runtime.authenticate_api_key(supplied, model)
        if record is None:
            self._json(401, {"error": "invalid_api_key"})
            return None
        return record

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        origin = self.headers.get("Origin", "")
        if origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:"):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/static/orbit-logo.png":
                logo = resources.files("orbit").joinpath("static/orbit-logo.png").read_bytes()
                self._send(200, logo, "image/png")
            elif path == "/api/health":
                self._json(200, {"name": "orbit", "status": "ok", "local": True})
            elif path == "/api/system":
                runtime = self.server.runtime
                self._json(200, {
                    "local": True,
                    "minimum_memory_gb": 10,
                    "data_root": str(runtime.data_root),
                    "presets": runtime.preset_rows(),
                    "training": runtime.training_state(),
                    "models": runtime.list_models(),
                    "active_model": runtime.active_model_id,
                    "loading": runtime.loading_state(),
                    "resources": runtime.system_state(),
                    "local_api_key": runtime.local_api_key,
                    "api_keys": runtime.list_api_keys(),
                })
            elif path == "/api/training":
                self._json(200, self.server.runtime.training_state())
            elif path == "/api/models":
                self._json(200, self.server.runtime.list_models())
            elif path == "/api/models/loading":
                self._json(200, self.server.runtime.loading_state())
            elif path == "/api/keys":
                self._json(200, self.server.runtime.list_api_keys())
            elif path == "/api/jobs":
                self._json(200, self._job_rows())
            elif path.startswith("/api/jobs/") and path.endswith("/download"):
                self._download_job(path.split("/")[3])
            elif path == "/v1/models":
                key = self._require_api_key()
                if not key:
                    return
                models = self.server.runtime.list_models()
                if key["model"] != "*":
                    models = [row for row in models if row["id"] == key["model"]]
                self._json(200, {
                    "object": "list",
                    "data": [
                        {"id": row["id"], "object": "model", "created": 0, "owned_by": "local-user"}
                        for row in models
                    ],
                })
            else:
                self._error(404, "not found")
        except Exception as exc:
            self._error(500, exc)

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            data = self._read_json()
            if path == "/api/jobs":
                self._create_job(data)
            elif path == "/api/training/start":
                self._json(202, self.server.runtime.start_training(data))
            elif path == "/api/training/auto":
                self._json(202, self.server.runtime.start_auto_training(data))
            elif path == "/api/training/stop":
                self._json(202, self.server.runtime.stop_training())
            elif path == "/api/models/load":
                self._json(202, self.server.runtime.start_load_model(str(data.get("model", ""))))
            elif path == "/api/models/unload":
                self._json(200, self.server.runtime.unload_model())
            elif path == "/api/keys/create":
                self._json(201, self.server.runtime.create_api_key(str(data.get("name", "")), str(data.get("model", "*"))))
            elif path == "/api/keys/revoke":
                self._json(200, self.server.runtime.revoke_api_key(str(data.get("id", ""))))
            elif path == "/api/chat":
                result = self.server.runtime.chat(
                    str(data.get("prompt", "")),
                    str(data["model"]) if data.get("model") else None,
                    int(data.get("max_tokens", 128)),
                    float(data.get("temperature", 0.8)),
                )
                self._json(200, result)
            elif path == "/v1/chat/completions":
                if not self._require_api_key(str(data.get("model", "")) or None):
                    return
                self._openai_chat(data)
            elif path == "/v1/responses":
                if not self._require_api_key(str(data.get("model", "")) or None):
                    return
                self._openai_response(data)
            else:
                self._error(404, "not found")
        except (ValueError, FileNotFoundError, MemoryError, RuntimeError) as exc:
            self._error(400, exc)
        except Exception as exc:
            self._error(500, exc)

    def _job_rows(self) -> list[dict[str, Any]]:
        rows = []
        for file in sorted(self.server.runtime.jobs_root.glob("*/job.json"), reverse=True):
            try:
                rows.append(json.loads(file.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return rows

    def _create_job(self, data: dict[str, Any]) -> None:
        train_cfg = TrainingConfig(
            steps=int(data.get("steps", 1000)),
            batch_size=int(data.get("batch_size", 1)),
            seq_len=int(data.get("seq_len", 2048)),
            grad_accum=int(data.get("grad_accum", 8)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            warmup_steps=int(data.get("warmup_steps", 100)),
            weight_decay=float(data.get("weight_decay", 0.1)),
            grad_clip=float(data.get("grad_clip", 1.0)),
            precision=str(data.get("precision", "auto")),
            scheduler=str(data.get("scheduler", "cosine")),
            checkpoint_every=int(data.get("checkpoint_every", 500)),
            seed=int(data.get("seed", 42)),
        )
        train_cfg.validate()
        text = str(data.get("text", "")).strip() or ("Orbit training sample. " * 100)
        zip_path = create_job_bundle(
            self.server.runtime.jobs_root,
            str(data.get("preset", "1b")),
            train_cfg.steps,
            train_cfg.batch_size,
            train_cfg.seq_len,
            train_cfg.learning_rate,
            text,
            training_config=train_cfg,
        )
        job_id = zip_path.stem.removeprefix("orbit-training-")
        self._json(201, {
            "job_id": job_id,
            "download": f"/api/jobs/{job_id}/download",
            "path": str(zip_path),
        })

    def _download_job(self, job_id: str) -> None:
        if Path(job_id).name != job_id:
            self._error(400, "invalid job id")
            return
        matches = list(self.server.runtime.jobs_root.glob(f"*-{job_id}.zip"))
        if not matches:
            self._error(404, "job not found")
            return
        data = matches[0].read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{matches[0].name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _openai_chat(self, data: dict[str, Any]) -> None:
        messages = data.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 必须是非空数组")
        parts = []
        prompt_tokens = 0
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("messages 中的每一项必须是对象")
            content = item.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            role = str(item.get("role", "user"))
            text = str(content)
            prompt_tokens += len(text.encode("utf-8"))
            parts.append(f"{role}: {text}")
        prompt = "\n".join(parts) + "\nassistant:"
        result = self.server.runtime.chat(
            prompt,
            str(data["model"]) if data.get("model") else None,
            int(data.get("max_tokens", 128)),
            float(data.get("temperature", 0.8)),
        )
        content = result["content"]
        now = int(time.time())
        self._json(200, {
            "id": f"chatcmpl-orbit-{now}",
            "object": "chat.completion",
            "created": now,
            "model": result["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(content.encode("utf-8")),
                "total_tokens": prompt_tokens + len(content.encode("utf-8")),
            },
        })

    def _openai_response(self, data: dict[str, Any]) -> None:
        raw_input = data.get("input", "")
        if isinstance(raw_input, str):
            prompt = raw_input
        elif isinstance(raw_input, list):
            prompt = "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in raw_input)
        else:
            raise ValueError("input 必须是字符串或数组")
        result = self.server.runtime.chat(
            prompt, str(data["model"]) if data.get("model") else None,
            int(data.get("max_output_tokens", 128)), float(data.get("temperature", 0.8)),
        )
        now = int(time.time())
        content = result["content"]
        self._json(200, {
            "id": f"resp_orbit_{now}", "object": "response", "created_at": now,
            "status": "completed", "model": result["model"],
            "output": [{"id": f"msg_orbit_{now}", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": content, "annotations": []}]}],
            "output_text": content,
        })


def _existing_server(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=0.7) as response:
            data = json.loads(response.read())
            return response.status == 200 and data.get("name") == "orbit"
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbit local training, chat and OpenAI-compatible API")
    parser.add_argument("--host", default="127.0.0.1", help="default is local-only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".orbit")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{args.port}"
    if _existing_server(url):
        print(f"Orbit 已在运行：{url}")
        if not args.no_browser:
            webbrowser.open(url)
        return

    memory_gb = OrbitConfig.system_memory_gb()
    if memory_gb > 0 and memory_gb <= 10:
        parser.error(f"Orbit requires more than 10 GB of memory; this computer reports about {memory_gb:.1f} GB")

    runtime = OrbitRuntime(args.data_dir)
    server = OrbitHTTPServer((args.host, args.port), runtime)
    server.verbose = args.verbose
    actual_port = server.server_address[1]
    url = f"http://{browser_host}:{actual_port}"
    print(f"Orbit 已启动：{url}")
    print(f"本机 API：{url}/v1")
    print(f"模型目录：{runtime.models_root}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nOrbit 已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
