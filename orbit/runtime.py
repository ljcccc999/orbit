from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import torch

from .config import OrbitConfig
from .model import OrbitForCausalLM
from .train import run_training, select_device
from .training_config import TrainingConfig


PRESETS = ("300m", "1b", "3b", "7b", "14b", "38b")


class OrbitRuntime:
    """Owns local training state and the model served by the web/API process."""

    def __init__(self, data_root: Path):
        self.data_root = data_root.expanduser().resolve()
        self.models_root = self.data_root / "models"
        self.jobs_root = self.data_root / "jobs"
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self._state_lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._training_thread: threading.Thread | None = None
        self._model: OrbitForCausalLM | None = None
        self._model_id: str | None = None
        self._training: dict[str, Any] = {
            "status": "idle",
            "step": 0,
            "steps": 0,
            "loss": None,
            "message": "尚未开始训练",
            "model_id": None,
        }

    def preset_rows(self) -> list[dict[str, Any]]:
        rows = []
        for preset in PRESETS:
            cfg = OrbitConfig.for_preset(preset)
            check = cfg.memory_check()
            rows.append({
                "id": preset,
                "parameters": cfg.estimate_parameters(),
                "training_memory_gb": round(cfg.estimated_training_memory_gb(), 1),
                "system_memory_gb": round(float(check["system_gb"]), 1),
                "can_train_here": bool(check["can_train"]),
            })
        return rows

    def training_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._training)

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.models_root.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
            stat = path.stat()
            rows.append({
                "id": path.stem,
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
                "active": path.stem == self._model_id,
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

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset = str(payload.get("preset", "300m")).lower()
        if preset not in PRESETS and preset != "local":
            raise ValueError("不支持的模型规模")
        train_cfg = TrainingConfig(
            steps=int(payload.get("steps", 100)),
            batch_size=int(payload.get("batch_size", 1)),
            seq_len=int(payload.get("seq_len", 256)),
            grad_accum=int(payload.get("grad_accum", 1)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            warmup_steps=int(payload.get("warmup_steps", 10)),
            weight_decay=float(payload.get("weight_decay", 0.1)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
            precision=str(payload.get("precision", "auto")),
            scheduler=str(payload.get("scheduler", "cosine")),
            checkpoint_every=int(payload.get("checkpoint_every", 100)),
            seed=int(payload.get("seed", 42)),
        )
        train_cfg.validate()
        text = str(payload.get("text", "")).strip()
        if len(text.encode("utf-8")) < train_cfg.seq_len + 2:
            raise ValueError("训练文本长度必须大于序列长度")

        requested_device = str(payload.get("device", "auto"))
        device = select_device(requested_device)
        cfg = OrbitConfig.for_preset(preset)
        if device.type in {"cpu", "mps"}:
            check = cfg.memory_check()
            if not check["can_train"]:
                raise MemoryError(
                    f"{preset} 预计需要约 {check['required_gb']:.1f}GB 训练内存，"
                    f"当前电脑约 {check['system_gb']:.1f}GB。请使用更小模型或导出远程 GPU 任务。"
                )

        with self._state_lock:
            if self._training_thread and self._training_thread.is_alive():
                raise RuntimeError("已经有一个本机训练任务正在运行")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            model_id = f"orbit-{preset}-{stamp}"
            checkpoint = self.models_root / f"{model_id}.pt"
            self._stop_event.clear()
            self._training = {
                "status": "running",
                "step": 0,
                "steps": train_cfg.steps,
                "loss": None,
                "message": f"正在使用 {device.type} 训练 {preset}",
                "model_id": model_id,
                "checkpoint": str(checkpoint),
                "device": device.type,
            }

        def callback(step: int, loss: float) -> None:
            with self._state_lock:
                self._training.update(step=step, loss=loss, message=f"训练中：{step}/{train_cfg.steps}")

        def worker() -> None:
            try:
                run_training(
                    device_name=requested_device,
                    checkpoint=checkpoint,
                    text=text,
                    preset=preset,
                    callback=callback,
                    stop_event=self._stop_event,
                    training_config=train_cfg,
                )
                with self._state_lock:
                    stopped = self._stop_event.is_set()
                    self._training.update(
                        status="stopped" if stopped else "completed",
                        message="训练已停止，已保存当前 checkpoint" if stopped else "训练完成，模型已保存在本机",
                    )
            except Exception as exc:
                with self._state_lock:
                    self._training.update(status="failed", message=str(exc))

        self._training_thread = threading.Thread(target=worker, name="orbit-training", daemon=True)
        self._training_thread.start()
        return self.training_state()

    def stop_training(self) -> dict[str, Any]:
        with self._state_lock:
            if not self._training_thread or not self._training_thread.is_alive():
                raise RuntimeError("当前没有正在运行的训练任务")
            self._stop_event.set()
            self._training["message"] = "正在安全停止并保存 checkpoint"
        return self.training_state()

    def load_model(self, model_id: str) -> dict[str, Any]:
        checkpoint = self._checkpoint_for(model_id)
        with self._model_lock:
            if self._model_id == model_id and self._model is not None:
                return {"id": model_id, "status": "ready"}
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:  # torch < 2.6
                state = torch.load(checkpoint, map_location="cpu")
            cfg = OrbitConfig(**state["config"])
            model = OrbitForCausalLM(cfg)
            model.load_state_dict(state["model"])
            model.eval()
            self._model = model
            self._model_id = model_id
        return {"id": model_id, "status": "ready"}

    def chat(self, prompt: str, model_id: str | None = None, max_tokens: int = 128, temperature: float = 0.8) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("消息不能为空")
        if not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens 必须在 1 到 2048 之间")
        if model_id:
            self.load_model(model_id)
        elif self._model is None:
            models = self.list_models()
            if not models:
                raise RuntimeError("还没有本地模型，请先完成训练")
            self.load_model(models[0]["id"])

        with self._model_lock:
            assert self._model is not None
            encoded = prompt.encode("utf-8")[-self._model.cfg.max_seq_len :]
            ids = torch.tensor([list(encoded)], dtype=torch.long)
            result = self._model.generate(ids, max_new_tokens=max_tokens, temperature=temperature)
            generated = bytes(result[0, ids.shape[1] :].tolist()).decode("utf-8", errors="replace")
            return {"model": self._model_id, "content": generated}

    @property
    def active_model_id(self) -> str | None:
        return self._model_id
