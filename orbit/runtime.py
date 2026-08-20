from __future__ import annotations

import gc
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config import OrbitConfig
from .resources import memory_is_critical, require_checkpoint_load_capacity, require_training_capacity, resource_snapshot
from .teacher import TeacherConfig, generate_dataset
from .training_config import TrainingConfig


PRESETS = ("300m", "1b", "3b", "7b", "14b", "38b")


class OrbitRuntime:
    """Own jobs and lazily load heavyweight ML dependencies and weights."""

    def __init__(self, data_root: Path, idle_unload_seconds: int | None = None):
        self.data_root = data_root.expanduser().resolve()
        self.models_root = self.data_root / "models"
        self.jobs_root = self.data_root / "jobs"
        self.datasets_root = self.data_root / "datasets"
        for path in (self.models_root, self.jobs_root, self.datasets_root):
            path.mkdir(parents=True, exist_ok=True)
        self.api_keys_path = self.data_root / "api-keys.json"
        self._api_keys = self._load_or_create_api_keys()
        self._state_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._work_thread: threading.Thread | None = None
        self._training_process: subprocess.Popen[str] | None = None
        self._load_thread: threading.Thread | None = None
        self._model: Any = None
        self._model_id: str | None = None
        self._model_device = "cpu"
        self._last_model_use = 0.0
        self._stop_reason = ""
        configured_idle = int(os.environ.get("ORBIT_MODEL_IDLE_SECONDS", "300"))
        self.idle_unload_seconds = max(10, idle_unload_seconds or configured_idle)
        self._training: dict[str, Any] = {
            "status": "idle", "step": 0, "steps": 0, "loss": None,
            "message": "尚未开始训练", "model_id": None,
        }
        self._loading: dict[str, Any] = {
            "status": "idle", "progress": 0, "message": "没有正在加载的模型", "model_id": None,
        }
        threading.Thread(target=self._idle_janitor, name="orbit-model-janitor", daemon=True).start()

    def _load_or_create_api_keys(self) -> list[dict[str, str]]:
        if self.api_keys_path.is_file():
            try:
                rows = json.loads(self.api_keys_path.read_text(encoding="utf-8"))
                if isinstance(rows, list) and rows:
                    return rows
            except (OSError, json.JSONDecodeError):
                pass
        legacy = self.data_root / "api-key"
        value = legacy.read_text(encoding="utf-8").strip() if legacy.is_file() else ""
        if len(value) < 24:
            value = "orbit_" + secrets.token_urlsafe(32)
        rows = [{"id": secrets.token_hex(8), "name": "Default", "key": value, "model": "*", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}]
        self._save_api_keys(rows)
        legacy.unlink(missing_ok=True)
        return rows

    def _save_api_keys(self, rows: list[dict[str, str]]) -> None:
        temporary = self.api_keys_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.api_keys_path)

    @property
    def local_api_key(self) -> str:
        return self._api_keys[0]["key"]

    def list_api_keys(self) -> list[dict[str, str]]:
        with self._state_lock:
            return [dict(row) for row in self._api_keys]

    def create_api_key(self, name: str, model: str = "*") -> dict[str, str]:
        name = name.strip() or "API Key"
        if len(name) > 80:
            raise ValueError("API Key 名称不能超过 80 个字符")
        if model != "*":
            self._checkpoint_for(model)
        row = {
            "id": secrets.token_hex(8), "name": name,
            "key": "orbit_" + secrets.token_urlsafe(32), "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self._state_lock:
            self._api_keys.append(row)
            self._save_api_keys(self._api_keys)
        return dict(row)

    def revoke_api_key(self, key_id: str) -> dict[str, str]:
        with self._state_lock:
            if len(self._api_keys) <= 1:
                raise ValueError("至少需要保留一个本机 API Key")
            found = next((row for row in self._api_keys if row["id"] == key_id), None)
            if found is None:
                raise ValueError("找不到该 API Key")
            self._api_keys = [row for row in self._api_keys if row["id"] != key_id]
            self._save_api_keys(self._api_keys)
            return {"status": "revoked", "id": key_id}

    def authenticate_api_key(self, value: str, model: str | None = None) -> dict[str, str] | None:
        import hmac
        for row in self.list_api_keys():
            if hmac.compare_digest(value, row["key"]):
                if model and row["model"] not in {"*", model}:
                    return None
                return row
        return None

    def preset_rows(self) -> list[dict[str, Any]]:
        rows = []
        snapshot = resource_snapshot(self.data_root)
        for preset in PRESETS:
            cfg = OrbitConfig.for_preset(preset)
            rows.append({
                "id": preset,
                "parameters": cfg.estimate_parameters(),
                "training_memory_gb": round(cfg.estimated_training_memory_gb(), 1),
                "system_memory_gb": snapshot["memory_total_gb"],
                "available_memory_gb": snapshot["memory_available_gb"],
                "can_train_here": cfg.estimated_training_memory_gb() <= max(0, snapshot["memory_available_gb"] - 2),
            })
        return rows

    def system_state(self) -> dict[str, Any]:
        return {
            **resource_snapshot(self.data_root),
            "model_loaded": self._model is not None,
            "idle_unload_seconds": self.idle_unload_seconds,
            "heavy_runtime_loaded": "torch" in sys.modules,
            "inference_worker_running": bool(self._model is not None and self._model.poll() is None),
        }

    def training_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._training)

    def loading_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._loading)

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.models_root.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
            stat = path.stat()
            rows.append({
                "id": path.stem, "filename": path.name, "size_bytes": stat.st_size,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
                "active": path.stem == self.active_model_id,
            })
        return rows

    def _checkpoint_for(self, model_id: str) -> Path:
        safe_id = Path(model_id).name
        if safe_id != model_id or safe_id in {"", ".", ".."}:
            raise ValueError("无效的模型名称")
        path = self.models_root / f"{safe_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"找不到本地模型：{model_id}")
        return path

    @staticmethod
    def _training_config(payload: dict[str, Any]) -> TrainingConfig:
        config = TrainingConfig(
            steps=int(payload.get("steps", 100)), batch_size=int(payload.get("batch_size", 1)),
            seq_len=int(payload.get("seq_len", 256)), grad_accum=int(payload.get("grad_accum", 1)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            warmup_steps=int(payload.get("warmup_steps", 10)), weight_decay=float(payload.get("weight_decay", 0.1)),
            grad_clip=float(payload.get("grad_clip", 1.0)), precision=str(payload.get("precision", "auto")),
            scheduler=str(payload.get("scheduler", "cosine")),
            checkpoint_every=int(payload.get("checkpoint_every", 100)), seed=int(payload.get("seed", 42)),
        )
        config.validate()
        return config

    def _assert_idle(self) -> None:
        if self._work_thread and self._work_thread.is_alive():
            raise RuntimeError("已经有一个训练或数据生成任务正在运行")

    def _prepare_training(self, payload: dict[str, Any], text: str) -> tuple[str, TrainingConfig, Path, str]:
        preset = str(payload.get("preset", "300m")).lower()
        if preset not in PRESETS and preset != "local":
            raise ValueError("不支持的模型规模")
        train_cfg = self._training_config(payload)
        if len(text.encode("utf-8")) < train_cfg.seq_len + 2:
            raise ValueError("训练文本长度必须大于序列长度")
        cfg = OrbitConfig.for_preset(preset)
        require_training_capacity(cfg.estimated_training_memory_gb(), cfg.estimate_parameters(), self.models_root)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        model_id = f"orbit-{preset}-{stamp}"
        return preset, train_cfg, self.models_root / f"{model_id}.pt", model_id

    def _run_local_training(self, payload: dict[str, Any], text: str) -> None:
        self.unload_model()
        preset, train_cfg, checkpoint, model_id = self._prepare_training(payload, text)
        requested_device = str(payload.get("device", "auto"))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dataset = Path(str(payload.get("_dataset_path", ""))) if payload.get("_dataset_path") else self.datasets_root / f"training-input-{stamp}.txt"
        if not dataset.is_file():
            temporary_dataset = dataset.with_suffix(".tmp")
            temporary_dataset.write_text(text, encoding="utf-8")
            os.replace(temporary_dataset, dataset)
        job_path = self.datasets_root / f"training-job-{stamp}.json"
        job_path.write_text(json.dumps({
            "preset": preset, "device": requested_device, "checkpoint": str(checkpoint),
            "dataset": str(dataset), "training_config": train_cfg.__dict__,
        }, ensure_ascii=False), encoding="utf-8")
        with self._state_lock:
            self._training.update(
                status="running", step=0, steps=train_cfg.steps, loss=None,
                message=f"正在启动隔离训练进程：{preset}", model_id=model_id,
                checkpoint=str(checkpoint), device=requested_device, phase="training",
                dataset=str(dataset),
            )
        log_path = self.data_root / "logs" / "training-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "orbit.training_worker", "--job", str(job_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_handle,
            text=True, bufsize=1,
        )
        log_handle.close()
        self._training_process = process

        def guard() -> None:
            while process.poll() is None:
                if memory_is_critical() and not self._stop_event.is_set():
                    self._stop_reason = "内存已接近安全下限，Orbit 已自动停止并保存 checkpoint"
                    self._stop_event.set()
                if self._stop_event.wait(1):
                    try:
                        assert process.stdin is not None
                        process.stdin.write('{"command":"stop"}\n')
                        process.stdin.flush()
                    except (OSError, BrokenPipeError):
                        pass
                    return

        threading.Thread(target=guard, name="orbit-memory-guard", daemon=True).start()
        assert process.stdout is not None
        final_type = ""
        try:
            for line in process.stdout:
                event = json.loads(line)
                event_type = str(event.get("type", ""))
                if event_type == "ready":
                    with self._state_lock:
                        self._training.update(device=event.get("device"), message=f"正在使用 {event.get('device')} 训练 {preset}")
                elif event_type == "progress":
                    with self._state_lock:
                        self._training.update(step=int(event["step"]), loss=float(event["loss"]), message=f"训练中：{event['step']}/{train_cfg.steps}")
                elif event_type in {"completed", "stopped"}:
                    final_type = event_type
                elif event_type == "fatal":
                    raise RuntimeError(str(event.get("error", "隔离训练进程失败")))
            return_code = process.wait()
            if return_code != 0 and not final_type:
                raise RuntimeError(f"隔离训练进程异常退出（{return_code}）")
            stopped = final_type == "stopped" or self._stop_event.is_set()
            with self._state_lock:
                self._training.update(
                    status="stopped" if stopped else "completed",
                    message=(self._stop_reason or "训练已停止，已原子保存当前 checkpoint") if stopped else "训练完成，模型已保存在本机",
                )
        finally:
            self._training_process = None
            job_path.unlink(missing_ok=True)

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        self._training_config(payload)
        with self._state_lock:
            self._assert_idle()
            self._stop_event.clear()
            self._stop_reason = ""
            self._training.update(status="preparing", step=0, message="正在检查内存、磁盘和训练配置", phase="preparing")

        def worker() -> None:
            try:
                self._run_local_training(payload, text)
            except Exception as exc:
                with self._state_lock:
                    self._training.update(status="failed", message=str(exc))

        self._work_thread = threading.Thread(target=worker, name="orbit-training", daemon=True)
        self._work_thread.start()
        return self.training_state()

    def start_auto_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("acknowledge_cost") is not True:
            raise ValueError("请先确认训练目标会发送给所选 AI API，且提供商可能收取费用")
        api_key = str(payload.get("api_key", "")).strip()
        teacher = TeacherConfig(
            base_url=str(payload.get("teacher_base_url", "https://api.deepseek.com")),
            model=str(payload.get("teacher_model", "deepseek-v4-flash")),
            instruction=str(payload.get("instruction", "")), examples=int(payload.get("examples", 20)),
            language=str(payload.get("language", "中文")),
        )
        teacher.validate()
        self._training_config(payload)
        with self._state_lock:
            self._assert_idle()
            self._stop_event.clear()
            self._stop_reason = ""
            self._training = {
                "status": "generating", "phase": "generation", "step": 0, "steps": teacher.examples,
                "loss": None, "message": "正在调用教师 API 生成训练样本", "model_id": None,
                "teacher_model": teacher.model,
            }

        def generated(current: int, total: int) -> None:
            with self._state_lock:
                self._training.update(step=current, steps=total, message=f"正在生成训练样本：{current}/{total}")

        def worker() -> None:
            try:
                text, usage = generate_dataset(teacher, api_key, self._stop_event, generated)
                if self._stop_event.is_set():
                    raise InterruptedError("自动训练已停止")
                stamp = time.strftime("%Y%m%d-%H%M%S")
                dataset = self.datasets_root / f"teacher-{stamp}.txt"
                temporary = dataset.with_suffix(".tmp")
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, dataset)
                with self._state_lock:
                    self._training.update(message="样本生成完成，正在启动本机训练", usage=usage, dataset=str(dataset))
                payload["_dataset_path"] = str(dataset)
                self._run_local_training(payload, text)
                with self._state_lock:
                    self._training.update(usage=usage, dataset=str(dataset))
            except InterruptedError as exc:
                with self._state_lock:
                    self._training.update(status="stopped", message=str(exc))
            except Exception as exc:
                with self._state_lock:
                    self._training.update(status="failed", message=str(exc))

        self._work_thread = threading.Thread(target=worker, name="orbit-auto-training", daemon=True)
        self._work_thread.start()
        return self.training_state()

    def stop_training(self) -> dict[str, Any]:
        with self._state_lock:
            if not self._work_thread or not self._work_thread.is_alive():
                raise RuntimeError("当前没有正在运行的训练或数据生成任务")
            self._stop_reason = "用户已请求安全停止"
            self._stop_event.set()
            self._training.update(status="stopping", message="正在安全停止；若已开始训练，将原子保存 checkpoint")
        return self.training_state()

    def _set_loading(self, status: str, progress: int, message: str, model_id: str | None) -> None:
        with self._state_lock:
            self._loading = {"status": status, "progress": progress, "message": message, "model_id": model_id}

    def load_model(self, model_id: str) -> dict[str, Any]:
        checkpoint = self._checkpoint_for(model_id)
        with self._model_lock:
            if self._model_id == model_id and self._model is not None and self._model.poll() is None:
                self._last_model_use = time.monotonic()
                return {"id": model_id, "status": "ready", "progress": 100}
            self._set_loading("loading", 10, "正在检查 checkpoint", model_id)
            require_checkpoint_load_capacity(checkpoint.stat().st_size, self.models_root)
            self.unload_model()
            log_path = self.data_root / "logs" / "inference-worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-m", "orbit.inference_worker", "--checkpoint", str(checkpoint)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_handle,
                text=True, bufsize=1,
            )
            log_handle.close()
            self._model = process
            self._model_id = model_id
            assert process.stdout is not None
            while True:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("隔离推理进程在模型就绪前退出")
                event = json.loads(line)
                if event.get("type") == "progress":
                    self._set_loading("loading", int(event["progress"]), str(event["message"]), model_id)
                elif event.get("type") == "ready":
                    self._model_device = str(event.get("device", "cpu"))
                    self._last_model_use = time.monotonic()
                    self._set_loading("ready", 100, str(event["message"]), model_id)
                    break
                elif event.get("type") in {"fatal", "error"}:
                    raise RuntimeError(str(event.get("error", "模型加载失败")))
        return {"id": model_id, "status": "ready", "progress": 100}

    def start_load_model(self, model_id: str) -> dict[str, Any]:
        self._checkpoint_for(model_id)
        with self._state_lock:
            if self._load_thread and self._load_thread.is_alive():
                if self._loading.get("model_id") == model_id:
                    return self.loading_state()
                raise RuntimeError("另一个模型正在加载")
            self._set_loading("queued", 0, "等待加载", model_id)

        def worker() -> None:
            try:
                self.load_model(model_id)
            except Exception as exc:
                self.unload_model()
                self._set_loading("failed", 0, str(exc), model_id)

        self._load_thread = threading.Thread(target=worker, name="orbit-model-loader", daemon=True)
        self._load_thread.start()
        return self.loading_state()

    @staticmethod
    def _release_memory() -> None:
        gc.collect()

    def unload_model(self) -> dict[str, Any]:
        with self._model_lock:
            previous = self._model_id
            process = self._model
            self._model = None
            self._model_id = None
            self._model_device = "cpu"
            self._last_model_use = 0.0
            if process is not None and process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.write('{"command":"shutdown"}\n')
                        process.stdin.flush()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
        self._release_memory()
        self._set_loading("idle", 0, "模型权重已从内存卸载", None)
        return {"status": "unloaded", "previous_model": previous}

    def _idle_janitor(self) -> None:
        while True:
            time.sleep(min(30, max(2, self.idle_unload_seconds // 4)))
            if self._model is not None and self._model.poll() is None and self._last_model_use:
                if time.monotonic() - self._last_model_use >= self.idle_unload_seconds:
                    self.unload_model()

    def chat(self, prompt: str, model_id: str | None = None, max_tokens: int = 128, temperature: float = 0.8) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("消息不能为空")
        if not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens 必须在 1 到 2048 之间")
        if model_id:
            self.load_model(model_id)
        elif self.active_model_id is None:
            models = self.list_models()
            if not models:
                raise RuntimeError("还没有本地模型，请先完成训练")
            self.load_model(models[0]["id"])
        with self._model_lock:
            assert self._model is not None and self._model.poll() is None
            self._last_model_use = time.monotonic()
            assert self._model.stdin is not None and self._model.stdout is not None
            self._model.stdin.write(json.dumps({
                "command": "chat", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature,
            }, ensure_ascii=False) + "\n")
            self._model.stdin.flush()
            line = self._model.stdout.readline()
            if not line:
                raise RuntimeError("隔离推理进程意外退出")
            response = json.loads(line)
            if response.get("type") != "result":
                raise RuntimeError(str(response.get("error", "模型生成失败")))
            self._last_model_use = time.monotonic()
            return {"model": self._model_id, "content": str(response.get("content", ""))}

    @property
    def active_model_id(self) -> str | None:
        return self._model_id if self._model is not None and self._model.poll() is None else None
