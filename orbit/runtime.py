from __future__ import annotations

import gc
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config import OrbitConfig
from .community import CommunityStore
from .conversations import ConversationStore
from .hub import OrbitHubClient
from .jobs import create_job_bundle
from .memory import LongTermMemory
from .identity import (
    ORBIT_SYSTEM_PROMPT,
    ORBIT_TRAINING_ANCHOR,
    identity_challenge,
    identity_response,
)
from .settings import OrbitSettings
from .resources import TRAINING_MEMORY_RESERVE_GB, memory_is_critical, require_checkpoint_load_capacity, require_training_capacity, resource_snapshot
from .teacher import TeacherConfig, generate_dataset
from .training_config import TrainingConfig


PRESETS = ("300m", "1b", "3b", "7b", "14b", "38b")
# Kept as a compatibility alias for existing metadata and integrations.
ORBIT_IDENTITY = ORBIT_SYSTEM_PROMPT


class OrbitRuntime:
    """Own jobs and lazily load heavyweight ML dependencies and weights."""

    def __init__(self, data_root: Path, idle_unload_seconds: int | None = None):
        self.data_root = data_root.expanduser().resolve()
        self.models_root = self.data_root / "models"
        self.jobs_root = self.data_root / "jobs"
        self.datasets_root = self.data_root / "datasets"
        self.training_runs_root = self.data_root / "training-runs"
        self.exports_root = self.data_root / "exports"
        for path in (self.models_root, self.jobs_root, self.datasets_root, self.training_runs_root, self.exports_root):
            path.mkdir(parents=True, exist_ok=True)
        self.api_keys_path = self.data_root / "api-keys.json"
        self._api_keys = self._load_or_create_api_keys()
        self.teacher_settings_path = self.data_root / "teacher-api.json"
        self._teacher_settings = self._load_teacher_settings()
        self.community = CommunityStore(self.data_root)
        self.conversations = ConversationStore(self.data_root)
        self.memory = LongTermMemory(self.data_root)
        self.hub = OrbitHubClient(self.data_root)
        self.settings = OrbitSettings(self.data_root)
        self._hub_upload: dict[str, Any] = {"status": "idle", "progress": 0, "message": "尚未上传", "model": None}
        self._pending_training: tuple[dict[str, Any], str, dict[str, Any], str] | None = None
        self._memory_resume_thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._work_thread: threading.Thread | None = None
        self._training_process: subprocess.Popen[str] | None = None
        self._load_thread: threading.Thread | None = None
        self._load_cancel = threading.Event()
        self._model: Any = None
        self._model_id: str | None = None
        self._model_device = "cpu"
        self._last_model_use = 0.0
        self._stop_reason = ""
        self._delete_after_stop = False
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
            value = "sk-" + secrets.token_urlsafe(32)
        rows = [{"id": secrets.token_hex(8), "name": "Default", "key": value, "model": "*", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}]
        self._save_api_keys(rows)
        legacy.unlink(missing_ok=True)
        return rows

    def _save_api_keys(self, rows: list[dict[str, str]]) -> None:
        temporary = self.api_keys_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.api_keys_path)

    @staticmethod
    def _default_teacher_settings() -> dict[str, Any]:
        return {
            "active_provider": "deepseek",
            "active_profiles": {"deepseek": "", "custom": ""},
            "profiles": {"deepseek": [], "custom": []},
        }

    def _normalize_teacher_settings(self, value: Any) -> dict[str, Any]:
        settings = self._default_teacher_settings()
        if not isinstance(value, dict):
            return settings
        # Migrate Orbit 0.4.1's single-provider file without losing its key.
        if "profiles" not in value:
            settings["profiles"]["deepseek"] = [{
                "id": secrets.token_hex(8),
                "base_url": str(value.get("base_url", "https://api.deepseek.com")),
                "model": str(value.get("model", "deepseek-v4-flash")),
                "api_key": str(value.get("api_key", "")),
            }]
            settings["active_profiles"]["deepseek"] = settings["profiles"]["deepseek"][0]["id"]
            return settings
        profiles = value.get("profiles")
        if isinstance(profiles, dict):
            for provider, profile in profiles.items():
                if provider not in {"deepseek", "custom"} or not isinstance(profile, dict):
                    if provider not in {"deepseek", "custom"} or not isinstance(profile, list):
                        continue
                raw_rows = profile.get("entries", []) if isinstance(profile, dict) else profile
                if isinstance(profile, dict) and "api_key" in profile:
                    raw_rows = [profile]
                rows = []
                if isinstance(raw_rows, list):
                    for row in raw_rows:
                        if not isinstance(row, dict):
                            continue
                        rows.append({
                            "id": str(row.get("id", "")) or secrets.token_hex(8),
                            "base_url": str(row.get("base_url", "")),
                            "model": str(row.get("model", "")),
                            "api_key": str(row.get("api_key", "")),
                        })
                settings["profiles"][provider] = rows
        active = str(value.get("active_provider", "deepseek"))
        settings["active_provider"] = active if active in settings["profiles"] else "deepseek"
        active_profiles = value.get("active_profiles", {})
        if isinstance(active_profiles, dict):
            for provider, profile_id in active_profiles.items():
                if provider in settings["profiles"] and any(row["id"] == str(profile_id) for row in settings["profiles"][provider]):
                    settings["active_profiles"][provider] = str(profile_id)
        for provider, rows in settings["profiles"].items():
            if not settings["active_profiles"].get(provider) and rows:
                settings["active_profiles"][provider] = rows[0]["id"]
        return settings

    def _load_teacher_settings(self) -> dict[str, Any]:
        if self.teacher_settings_path.is_file():
            try:
                value = json.loads(self.teacher_settings_path.read_text(encoding="utf-8"))
                return self._normalize_teacher_settings(value)
            except (OSError, json.JSONDecodeError):
                pass
        return self._default_teacher_settings()

    def _save_teacher_settings(self, settings: dict[str, Any]) -> None:
        settings = self._normalize_teacher_settings(settings)
        temporary = self.teacher_settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.teacher_settings_path)
        self._teacher_settings = settings

    def teacher_settings(self) -> dict[str, Any]:
        with self._state_lock:
            return json.loads(json.dumps(self._teacher_settings))

    def public_teacher_settings(self) -> dict[str, Any]:
        settings = self.teacher_settings()
        public_profiles: dict[str, Any] = {}
        for provider, rows in settings["profiles"].items():
            public_profiles[provider] = {
                "active_id": settings["active_profiles"].get(provider, ""),
                "entries": [
                    {
                        "id": row["id"], "base_url": row["base_url"], "model": row["model"],
                        "has_api_key": bool(row.get("api_key")),
                        "key_hint": ("••••" + row["api_key"][-4:]) if row.get("api_key") else "",
                    }
                    for row in rows
                ],
            }
        settings["profiles"] = public_profiles
        return settings

    def save_teacher_profile(self, provider: str, base_url: str, model: str, api_key: str, profile_id: str = "", create_new: bool = False) -> dict[str, Any]:
        provider = provider.strip().lower()
        if provider not in {"deepseek", "custom"}:
            raise ValueError("不支持的教师 API 提供商")
        values = {"base_url": base_url.strip(), "model": model.strip(), "api_key": api_key.strip()}
        if any("\n" in value or len(value) > 1000 for value in values.values()):
            raise ValueError("教师 API 配置无效")
        with self._state_lock:
            settings = self.teacher_settings()
            rows = settings["profiles"].setdefault(provider, [])
            profile_id = profile_id.strip()
            selected = next((row for row in rows if row["id"] == profile_id), None)
            if selected is None and not create_new:
                active_id = settings["active_profiles"].get(provider, "")
                selected = next((row for row in rows if row["id"] == active_id), None)
            if selected is not None:
                if not values["api_key"]:
                    values["api_key"] = selected.get("api_key", "")
                selected.update(values)
                profile_id = selected["id"]
            else:
                if not values["api_key"]:
                    raise ValueError("请填写有效的 API Key")
                profile_id = secrets.token_hex(8)
                rows.append({"id": profile_id, **values})
            settings["active_provider"] = provider
            settings["active_profiles"][provider] = profile_id
            self._save_teacher_settings(settings)
            return self.public_teacher_settings()

    def select_teacher_profile(self, provider: str, profile_id: str) -> dict[str, Any]:
        provider = provider.strip().lower()
        with self._state_lock:
            settings = self.teacher_settings()
            rows = settings["profiles"].get(provider, [])
            if provider not in {"deepseek", "custom"} or not any(row["id"] == profile_id for row in rows):
                raise ValueError("找不到教师 API 配置")
            settings["active_provider"] = provider
            settings["active_profiles"][provider] = profile_id
            self._save_teacher_settings(settings)
            return self.public_teacher_settings()

    def delete_teacher_profile(self, provider: str, profile_id: str) -> dict[str, Any]:
        provider = provider.strip().lower()
        with self._state_lock:
            settings = self.teacher_settings()
            rows = settings["profiles"].get(provider, [])
            remaining = [row for row in rows if row["id"] != profile_id]
            if len(remaining) == len(rows):
                raise FileNotFoundError("找不到教师 API 配置")
            settings["profiles"][provider] = remaining
            active_id = settings["active_profiles"].get(provider, "")
            settings["active_profiles"][provider] = active_id if any(row["id"] == active_id for row in remaining) else (remaining[0]["id"] if remaining else "")
            self._save_teacher_settings(settings)
            return self.public_teacher_settings()

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
            "key": "sk-" + secrets.token_urlsafe(32), "model": model,
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
                "can_train_here": cfg.estimated_training_memory_gb() <= max(0, snapshot["memory_available_gb"] - TRAINING_MEMORY_RESERVE_GB),
            })
        return rows

    def training_recommendation(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset = str(payload.get("preset", "300m")).lower()
        if preset not in PRESETS:
            raise ValueError("不支持的模型规模")
        device = str(payload.get("device", "auto")).lower()
        if device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("不支持的训练设备")
        examples = max(1, min(100, int(payload.get("examples", 20))))
        text_chars = max(0, min(100_000_000, int(payload.get("text_chars", 0))))
        model = OrbitConfig.for_preset(preset)
        base = TrainingConfig.for_model(preset)
        resources = resource_snapshot(self.data_root)
        available = float(resources.get("memory_available_gb", 0) or 0)
        required = model.estimated_training_memory_gb()
        safe_budget = max(0.0, available - TRAINING_MEMORY_RESERVE_GB)
        feasible = required <= safe_budget
        pressure = required / safe_budget if safe_budget else float("inf")
        seq_len = base.seq_len
        if pressure > 0.75:
            seq_len = max(256, seq_len // 2)
        if device == "cpu":
            seq_len = min(seq_len, 1024)
        data_units = examples if bool(payload.get("assisted")) else max(1, text_chars // max(256, seq_len))
        scale_examples = {"300m": 20, "1b": 36, "3b": 56, "7b": 72, "14b": 88, "38b": 100}[preset]
        goal_chars = max(0, min(20_000, int(payload.get("goal_chars", 0))))
        recommended_examples = min(100, scale_examples + min(20, goal_chars // 250))
        steps = max(100, min(2000, data_units * (20 if bool(payload.get("assisted")) else 8)))
        warmup = max(10, min(200, steps // 10))
        checkpoint_every = max(25, min(250, steps // 5))
        config = base.with_overrides(
            steps=steps,
            batch_size=1 if pressure > 0.55 or device == "cpu" else base.batch_size,
            seq_len=seq_len,
            warmup_steps=warmup,
            checkpoint_every=checkpoint_every,
            precision="auto",
        )
        # Pre-training estimates are deliberately labeled as rough. Once a
        # run starts, training_state() replaces them with measured ETA from
        # completed optimizer steps. The baseline is calibrated to the local
        # 300M MPS path and scaled by parameter count and token work.
        parameter_scale = max(0.25, (model.estimate_parameters() / 308_450_304) ** 0.85)
        token_work = (config.seq_len / 512) * config.batch_size * (config.grad_accum / 8)
        device_factor = {"cpu": 2.0, "mps": 1.0, "cuda": 0.45}.get(device, 1.0)
        estimated_step_seconds = max(1.0, 300.0 * parameter_scale * token_work * device_factor)
        estimated_training_seconds = round(estimated_step_seconds * config.steps + 120)
        activation_ratio = (config.seq_len / max(1, base.seq_len)) * (config.batch_size / max(1, base.batch_size))
        estimated_peak_memory = max(required, required * (0.8 + 0.2 * max(0.25, activation_ratio)))
        return {
            "preset": preset,
            "device": device,
            "feasible": feasible,
            "reason": "local_memory" if not feasible else "balanced_for_device_and_data",
            "required_memory_gb": round(required, 1),
            "available_memory_gb": round(available, 1),
            "recommended_examples": recommended_examples,
            "config": config.__dict__,
            "estimated_step_seconds": round(estimated_step_seconds),
            "estimated_training_seconds": estimated_training_seconds,
            "estimated_peak_memory_gb": round(estimated_peak_memory, 1),
            "estimated_activation_delta_gb": round(max(0.0, estimated_peak_memory - required), 1),
            "estimate_note": "粗略估算；训练开始后会用实际步速和 ETA 替换。教师 API 生成时间另计。",
        }

    def system_state(self) -> dict[str, Any]:
        orbit_memory = None
        try:
            import psutil
            process = psutil.Process(os.getpid())
            orbit_memory = process.memory_info().rss
            orbit_memory += sum(child.memory_info().rss for child in process.children(recursive=True) if child.is_running())
        except Exception:
            pass
        return {
            **resource_snapshot(self.data_root),
            "orbit_memory_gb": round(orbit_memory / 1_000_000_000, 2) if orbit_memory is not None else None,
            "model_loaded": self._model is not None,
            "idle_unload_seconds": self.idle_unload_seconds,
            "heavy_runtime_loaded": "torch" in sys.modules,
            "inference_worker_running": bool(self._model is not None and self._model.poll() is None),
        }

    def training_state(self) -> dict[str, Any]:
        with self._state_lock:
            state = dict(self._training)
        started = float(state.pop("_started_monotonic", 0) or 0)
        elapsed = max(0.0, time.monotonic() - started) if started else 0.0
        state["elapsed_seconds"] = round(elapsed)
        step, steps = int(state.get("step", 0) or 0), int(state.get("steps", 0) or 0)
        state["eta_seconds"] = round(elapsed / step * (steps - step)) if started and step > 0 and steps > step else None
        return state

    def loading_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._loading)

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.models_root.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
            stat = path.stat()
            metadata = self._model_metadata(path.stem)
            rows.append({
                "id": path.stem, "filename": path.name, "size_bytes": stat.st_size,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
                "active": path.stem == self.active_model_id,
                **metadata,
            })
        return rows

    def check_model_name(self, name: str) -> dict[str, Any]:
        """Check a custom model name without creating files or changing state."""
        raw = str(name or "").strip()
        if not raw:
            return {"valid": True, "duplicate": False, "name": "", "message": ""}
        try:
            normalized = self._safe_model_name(raw, raw)
        except ValueError as exc:
            return {"valid": False, "duplicate": False, "name": raw, "message": str(exc)}
        duplicate = any((self.models_root / f"{normalized}{suffix}").is_file() for suffix in (".pt", ".json"))
        return {
            "valid": not duplicate,
            "duplicate": duplicate,
            "name": normalized,
            "message": (f"模型名称已存在：{normalized}，请换一个名称" if duplicate else "名称可用"),
        }

    @staticmethod
    def _safe_model_name(value: str, fallback: str) -> str:
        value = value.strip()
        if not value:
            return fallback
        value = "-".join(value.split())
        value = "".join(ch for ch in value if ch.isalnum() or ch in "._-").strip(".-_")
        if not value or len(value) > 80:
            raise ValueError("模型名称需要是 1–80 个字母、数字、中文、点、横线或下划线")
        return value

    def _metadata_path(self, model_id: str) -> Path:
        return self.models_root / f"{model_id}.json"

    def _model_metadata(self, model_id: str) -> dict[str, Any]:
        path = self._metadata_path(model_id)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Metadata can come from an older checkpoint or a user
                    # edited file. The product identity is never user-trained.
                    data["identity"] = "Orbit"
                    data["developer"] = "YUNSH"
                    data["system_prompt"] = ORBIT_IDENTITY
                    data["identity_training_examples"] = ORBIT_TRAINING_ANCHOR
                    data["ollama_ready"] = bool(data.get("ollama_ready")) or (self.models_root / f"{model_id}.gguf").is_file()
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        preset = next((name for name in PRESETS if re.search(rf"(^|[-_]){re.escape(name)}($|[-_])", model_id.lower())), "custom")
        parameters = OrbitConfig.for_preset(preset).estimate_parameters() if preset in PRESETS else None
        return {
            "name": model_id, "preset": preset, "parameters": parameters,
            "identity": "Orbit", "developer": "YUNSH", "system_prompt": ORBIT_IDENTITY,
            "identity_training_examples": ORBIT_TRAINING_ANCHOR,
            "parent_model": None, "training_runs": [], "architecture": "orbit-hybrid-moe-v1",
            "ollama_ready": (self.models_root / f"{model_id}.gguf").is_file(),
        }

    def _write_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> None:
        path = self._metadata_path(model_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def list_training_runs(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.training_runs_root.glob("*/run.json"), reverse=True):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                row.pop("content", None)
                rows.append(row)
            except (OSError, json.JSONDecodeError):
                continue
        return rows

    def training_run(self, run_id: str) -> dict[str, Any]:
        if Path(run_id).name != run_id:
            raise ValueError("无效的训练记录")
        path = self.training_runs_root / run_id / "run.json"
        if not path.is_file():
            raise FileNotFoundError("找不到训练记录")
        row = json.loads(path.read_text(encoding="utf-8"))
        dataset = Path(str(row.get("dataset", "")))
        row["content"] = dataset.read_text(encoding="utf-8") if dataset.is_file() else ""
        training_dataset = Path(str(row.get("training_dataset", "")))
        row["training_content"] = training_dataset.read_text(encoding="utf-8") if training_dataset.is_file() else ""
        return row

    def delete_training_run(self, run_id: str) -> dict[str, str]:
        if Path(run_id).name != run_id or not run_id:
            raise ValueError("无效的训练记录")
        with self._state_lock:
            if self._training.get("run_id") == run_id and self._work_thread and self._work_thread.is_alive():
                raise RuntimeError("训练正在运行，请先停止训练后再删除历史")
        root = self.training_runs_root / run_id
        path = root / "run.json"
        if not path.is_file():
            raise FileNotFoundError("找不到训练记录")
        row = json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset", "training_dataset"):
            candidate = Path(str(row.get(key, "")))
            try:
                if candidate.is_file() and candidate.resolve().parent == self.datasets_root.resolve():
                    candidate.unlink()
            except OSError:
                pass
        shutil.rmtree(root)
        return {"status": "deleted", "id": run_id}

    def _save_run(self, run: dict[str, Any]) -> None:
        root = self.training_runs_root / str(run["id"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / "run.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

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

    @staticmethod
    def _data_language(payload: dict[str, Any]) -> str:
        language = str(payload.get("data_language", "bilingual"))
        if language not in {"zh", "en", "bilingual"}:
            raise ValueError("训练语言必须是中文、English 或中英双语")
        return language

    def _teacher_model_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset = str(payload.get("preset", "300m")).lower()
        cfg = OrbitConfig.for_preset(preset)
        train_cfg = self._training_config(payload)
        return {
            "preset": preset,
            "parameters": cfg.estimate_parameters(),
            "identity": "Orbit",
            "developer": "YUNSH",
            "identity_training_rule": "回答‘你是谁’时必须说明自己是 Orbit，由 YUNSH 开发；训练内容不能改变产品身份。",
            "context_length": train_cfg.seq_len,
            "training_steps": train_cfg.steps,
            "base_model": str(payload.get("base_model", "")).strip() or None,
            "model_name": str(payload.get("model_name", "")).strip() or None,
            "data_language": self._data_language(payload),
        }

    def _assert_idle(self) -> None:
        if self._work_thread and self._work_thread.is_alive():
            raise RuntimeError("已经有一个训练或数据生成任务正在运行")

    def _prepare_training(self, payload: dict[str, Any], text: str) -> tuple[str, TrainingConfig, Path, str, str, str | None]:
        preset = str(payload.get("preset", "300m")).lower()
        if preset not in PRESETS and preset != "local":
            raise ValueError("不支持的模型规模")
        train_cfg = self._training_config(payload)
        self._data_language(payload)
        if len(text.encode("utf-8")) < train_cfg.seq_len + 2:
            raise ValueError("训练文本长度必须大于序列长度")
        cfg = OrbitConfig.for_preset(preset)
        require_training_capacity(cfg.estimated_training_memory_gb(), cfg.estimate_parameters(), self.models_root)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        parent_model = str(payload.get("base_model", "")).strip() or None
        if parent_model:
            self._checkpoint_for(parent_model)
            parent_preset = str(self._model_metadata(parent_model).get("preset", ""))
            if parent_preset and parent_preset != preset:
                raise ValueError(f"二次训练必须保持父模型规模：请选择 {parent_preset.upper()}")
        resume_model_id = str(payload.get("_resume_model_id", "")).strip()
        if resume_model_id:
            checkpoint = self._checkpoint_for(resume_model_id)
            metadata = self._model_metadata(resume_model_id)
            display_name = str(metadata.get("display_name") or metadata.get("name") or resume_model_id)
            return preset, train_cfg, checkpoint, resume_model_id, display_name, parent_model
        fallback = f"orbit-{preset}-{stamp}"
        requested_name = str(payload.get("model_name", "")).strip()
        display_name = self._safe_model_name(requested_name, fallback)
        model_id = display_name
        checkpoint = self.models_root / f"{model_id}.pt"
        metadata_path = self._metadata_path(model_id)
        if checkpoint.exists() or metadata_path.exists():
            if requested_name:
                raise ValueError(f"模型名称已存在：{display_name}，请换一个名称")
            suffix = 2
            while checkpoint.exists() or metadata_path.exists():
                model_id = f"{display_name}-{suffix}"
                checkpoint = self.models_root / f"{model_id}.pt"
                metadata_path = self._metadata_path(model_id)
                suffix += 1
            display_name = model_id
        return preset, train_cfg, checkpoint, model_id, display_name, parent_model

    def _run_local_training(self, payload: dict[str, Any], text: str) -> None:
        self.unload_model()
        preset, train_cfg, checkpoint, model_id, display_name, parent_model = self._prepare_training(payload, text)
        requested_device = str(payload.get("device", "auto"))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dataset = Path(str(payload.get("_dataset_path", ""))) if payload.get("_dataset_path") else self.datasets_root / f"training-input-{stamp}.txt"
        if not dataset.is_file():
            temporary_dataset = dataset.with_suffix(".tmp")
            temporary_dataset.write_text(text, encoding="utf-8")
            os.replace(temporary_dataset, dataset)
        training_dataset = self.datasets_root / f"training-corpus-{stamp}-{secrets.token_hex(3)}.txt"
        training_text = text if ORBIT_TRAINING_ANCHOR in text else f"{ORBIT_TRAINING_ANCHOR}\n\n{text}\n\n{ORBIT_TRAINING_ANCHOR}\n"
        training_dataset.write_text(training_text, encoding="utf-8")
        resume_run_id = str(payload.get("_resume_run_id", "")).strip()
        run_id = resume_run_id or f"{stamp}-{secrets.token_hex(3)}"
        cfg = OrbitConfig.for_preset(preset)
        existing_run: dict[str, Any] = {}
        if resume_run_id:
            run_path = self.training_runs_root / resume_run_id / "run.json"
            if run_path.is_file():
                try:
                    loaded = json.loads(run_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing_run = loaded
                except (OSError, json.JSONDecodeError):
                    pass
        run = {
            **existing_run,
            "id": run_id, "status": "running", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "completed_at": None, "model_id": model_id, "model_name": display_name,
            "preset": preset, "parameters": cfg.estimate_parameters(), "parent_model": parent_model,
            "assisted": bool(payload.get("_assisted")), "training_goal": str(payload.get("instruction", "")),
            "identity_training_injected": True,
            "dataset": str(dataset), "training_dataset": str(training_dataset),
            "training_config": train_cfg.__dict__, "device": requested_device,
            "step": int(existing_run.get("step", 0) or 0), "steps": train_cfg.steps,
            "loss": existing_run.get("loss"), "message": "正在继续训练" if resume_run_id else "正在启动训练",
            "data_language": self._data_language(payload),
        }
        self._save_run(run)
        metadata = {
            "name": display_name, "display_name": display_name, "model_id": model_id,
            "preset": preset, "parameters": cfg.estimate_parameters(),
            "identity": "Orbit", "developer": "YUNSH", "system_prompt": ORBIT_IDENTITY, "parent_model": parent_model,
            "identity_training_examples": ORBIT_TRAINING_ANCHOR,
            "training_runs": [run_id], "architecture": "orbit-hybrid-moe-v1", "ollama_ready": False,
            "created_at": run["created_at"],
        }
        if parent_model:
            parent_metadata = self._model_metadata(parent_model)
            metadata["training_runs"] = [*parent_metadata.get("training_runs", []), run_id]
        job_path = self.datasets_root / f"training-job-{stamp}.json"
        resume_checkpoint = str(payload.get("_resume_checkpoint", "")).strip()
        if not resume_checkpoint and parent_model:
            resume_checkpoint = str(self._checkpoint_for(parent_model))
        job_path.write_text(json.dumps({
            "preset": preset, "device": requested_device, "checkpoint": str(checkpoint),
            "dataset": str(training_dataset), "training_config": train_cfg.__dict__,
            "resume": resume_checkpoint or None,
            "resume_weights_only": bool(parent_model and not resume_checkpoint),
            "metadata": metadata,
        }, ensure_ascii=False), encoding="utf-8")
        with self._state_lock:
            self._training.update(
                status="running", step=int(existing_run.get("step", 0) or 0), steps=train_cfg.steps,
                loss=existing_run.get("loss"),
                message=(f"正在继续隔离训练进程：{preset}" if resume_run_id else f"正在启动隔离训练进程：{preset}"), model_id=model_id, model_name=display_name,
                checkpoint=str(checkpoint), device=requested_device, phase="training",
                dataset=str(dataset), run_id=run_id, parent_model=parent_model,
                _started_monotonic=time.monotonic(),
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

        memory_stop = False

        def guard() -> None:
            nonlocal memory_stop
            while process.poll() is None:
                if memory_is_critical() and not self._stop_event.is_set():
                    memory_stop = True
                    self._stop_reason = "内存已接近安全下限，Orbit 已自动停止并保存 checkpoint"
                    self._stop_event.set()
                if self._stop_event.wait(1):
                    try:
                        assert process.stdin is not None
                        process.stdin.write('{"command":"stop"}\n')
                        process.stdin.flush()
                    except (OSError, BrokenPipeError):
                        pass
                    # MPS or a native kernel can occasionally stop servicing
                    # the worker's stdin thread. Never leave the UI in
                    # "Stopping" forever: give the cooperative stop a short
                    # grace period, then terminate only this training child.
                    deadline = time.monotonic() + 5
                    while process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.2)
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
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
                    run.update(step=int(event["step"]), loss=float(event["loss"]), message=self._training["message"])
                    self._save_run(run)
                elif event_type in {"completed", "stopped"}:
                    final_type = event_type
                elif event_type == "fatal":
                    raise RuntimeError(str(event.get("error", "隔离训练进程失败")))
            return_code = process.wait()
            if return_code != 0 and not final_type and not self._stop_event.is_set():
                raise RuntimeError(f"隔离训练进程异常退出（{return_code}）")
            stopped = final_type == "stopped" or self._stop_event.is_set()
            with self._state_lock:
                self._training.update(
                    status="stopped" if stopped else "completed",
                    message=(self._stop_reason or "训练已停止，已原子保存当前 checkpoint") if stopped else "训练完成，模型已保存在本机",
                )
            waiting_for_memory = stopped and memory_stop and not self._delete_after_stop
            run.update(
                status="stopped_deleted" if stopped and self._delete_after_stop else ("waiting_memory" if waiting_for_memory else ("stopped" if stopped else "completed")),
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                step=int(self._training.get("step", 0)), loss=self._training.get("loss"),
                message=("训练已暂停，内存恢复到安全水平后会自动继续" if waiting_for_memory else ("训练已停止，未完成模型已删除" if stopped and self._delete_after_stop else self._training.get("message"))),
            )
            if stopped and self._delete_after_stop:
                self._remove_model_files(model_id)
                with self._state_lock:
                    self._training.update(status="stopped_deleted", message="训练已停止，未完成模型已删除")
            self._save_run(run)
            if not (stopped and self._delete_after_stop):
                self._write_model_metadata(model_id, metadata)
            if waiting_for_memory:
                resume_payload = dict(payload)
                resume_payload["_resume_model_id"] = model_id
                resume_payload["_resume_checkpoint"] = str(checkpoint)
                resume_payload["_resume_run_id"] = run_id
                with self._state_lock:
                    self._pending_training = (resume_payload, text, {}, str(dataset))
                    self._training.update(
                        status="waiting_memory",
                        message=f"训练已暂停，保留 {TRAINING_MEMORY_RESERVE_GB:.0f}GB 安全内存后会自动继续",
                    )
                self._start_memory_resume_monitor()
        except Exception as exc:
            run.update(status="failed", completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), message=str(exc))
            self._save_run(run)
            raise
        finally:
            self._training_process = None
            self._delete_after_stop = False
            job_path.unlink(missing_ok=True)

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        self._training_config(payload)
        with self._state_lock:
            self._assert_idle()
            self._stop_event.clear()
            self._stop_reason = ""
            self._delete_after_stop = False
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
        provider = str(payload.get("teacher_provider", "deepseek")).strip().lower()
        private_settings = self.teacher_settings()
        stored_rows = private_settings.get("profiles", {}).get(provider, [])
        requested_profile_id = str(payload.get("teacher_profile_id", "")).strip()
        active_profile_id = private_settings.get("active_profiles", {}).get(provider, "")
        stored_profile = next((row for row in stored_rows if row.get("id") == (requested_profile_id or active_profile_id)), {})
        api_key = str(payload.get("api_key", "")).strip() or str(stored_profile.get("api_key", "")).strip()
        teacher = TeacherConfig(
            base_url=str(payload.get("teacher_base_url", "https://api.deepseek.com")),
            model=str(payload.get("teacher_model", "deepseek-v4-flash")),
            instruction=str(payload.get("instruction", "")), examples=int(payload.get("examples", 20)),
            language=str(payload.get("language", "中文")),
            model_profile=self._teacher_model_profile(payload),
        )
        teacher.validate()
        self._training_config(payload)
        if not api_key or "\n" in api_key or len(api_key) > 1000:
            raise ValueError("请填写有效的 API Key")
        self.save_teacher_profile(provider, teacher.base_url, teacher.model, api_key, profile_id=requested_profile_id or active_profile_id)
        with self._state_lock:
            self._assert_idle()
            self._stop_event.clear()
            self._stop_reason = ""
            self._delete_after_stop = False
            self._training = {
                "status": "generating", "phase": "generation", "step": 0, "steps": teacher.examples,
                "loss": None, "message": "正在调用教师 API 生成训练样本", "model_id": None,
                "teacher_model": teacher.model,
                "_started_monotonic": time.monotonic(),
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
                    self._training.update(status="waiting_memory", message="样本已保存，正在等待足够的安全训练内存", usage=usage, dataset=str(dataset))
                payload["_dataset_path"] = str(dataset)
                payload["_assisted"] = True
                preset = str(payload.get("preset", "300m")).lower()
                required = OrbitConfig.for_preset(preset).estimated_training_memory_gb()
                available = float(resource_snapshot(self.data_root).get("memory_available_gb", 0) or 0)
                if required > max(0.0, available - TRAINING_MEMORY_RESERVE_GB):
                    with self._state_lock:
                        self._pending_training = (dict(payload), text, usage, str(dataset))
                        self._training.update(
                            status="needs_memory",
                            message=f"内存不足，训练没有卡住：样本已保存。训练需要约 {required:.1f}GB，当前可用 {available:.1f}GB；至少保留 {TRAINING_MEMORY_RESERVE_GB:.0f}GB 后会自动继续",
                        )
                    self._start_memory_resume_monitor()
                    return
                with self._state_lock:
                    self._training.update(status="preparing", message="内存已满足要求，正在启动本机训练")
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

    def resume_pending_training(self) -> dict[str, Any]:
        with self._state_lock:
            pending = self._pending_training
            if not pending:
                raise RuntimeError("没有等待继续的 AI 辅助训练")
            payload, text, usage, dataset = pending
            preset = str(payload.get("preset", "300m")).lower()
            required = OrbitConfig.for_preset(preset).estimated_training_memory_gb()
            available = float(resource_snapshot(self.data_root).get("memory_available_gb", 0) or 0)
            if required > max(0.0, available - TRAINING_MEMORY_RESERVE_GB):
                raise RuntimeError(f"内存仍不足：训练约需 {required:.1f}GB，当前可用 {available:.1f}GB，并需保留 {TRAINING_MEMORY_RESERVE_GB:.0f}GB 给系统")
            self._assert_idle()
            self._pending_training = None
            self._stop_event.clear()
            self._training.update(status="preparing", message="内存已满足要求，正在继续训练")

        def worker() -> None:
            try:
                self._run_local_training(payload, text)
                with self._state_lock:
                    self._training.update(usage=usage, dataset=dataset)
            except Exception as exc:
                with self._state_lock:
                    self._training.update(status="failed", message=str(exc))

        self._work_thread = threading.Thread(target=worker, name="orbit-resumed-training", daemon=True)
        self._work_thread.start()
        return self.training_state()

    def _start_memory_resume_monitor(self) -> None:
        with self._state_lock:
            if self._memory_resume_thread and self._memory_resume_thread.is_alive():
                return
            self._memory_resume_thread = threading.Thread(
                target=self._memory_resume_loop, name="orbit-memory-resume", daemon=True,
            )
            self._memory_resume_thread.start()

    def _memory_resume_loop(self) -> None:
        while True:
            with self._state_lock:
                pending = self._pending_training
                worker = self._work_thread
                status = str(self._training.get("status", ""))
            if not pending or status not in {"waiting_memory", "needs_memory"}:
                return
            if worker and worker.is_alive():
                time.sleep(1)
                continue
            payload, _text, _usage, _dataset = pending
            try:
                preset = str(payload.get("preset", "300m")).lower()
                required = OrbitConfig.for_preset(preset).estimated_training_memory_gb()
                available = float(resource_snapshot(self.data_root).get("memory_available_gb", 0) or 0)
                if required <= max(0.0, available - TRAINING_MEMORY_RESERVE_GB):
                    self.resume_pending_training()
                    return
            except Exception as exc:
                with self._state_lock:
                    self._training.update(status="failed", message=str(exc))
                return
            time.sleep(3)

    def continue_training(self, run_id: str | None = None) -> dict[str, Any]:
        """Resume the interrupted run from its existing checkpoint."""
        with self._state_lock:
            self._assert_idle()
            state = dict(self._training)
        selected_id = str(run_id or state.get("run_id") or "").strip()
        if not selected_id or Path(selected_id).name != selected_id:
            raise ValueError("找不到可继续的训练记录")
        run_path = self.training_runs_root / selected_id / "run.json"
        if not run_path.is_file():
            raise FileNotFoundError("找不到训练记录")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        model_id = str(run.get("model_id") or state.get("model_id") or "").strip()
        checkpoint = self._checkpoint_for(model_id)
        dataset = Path(str(run.get("dataset", "")))
        if not dataset.is_file():
            raise FileNotFoundError("找不到这次训练保存的内容")
        config = dict(run.get("training_config") or {})
        payload: dict[str, Any] = {
            "model_name": str(run.get("model_name") or model_id),
            "preset": str(run.get("preset", "300m")),
            "base_model": "", "device": str(run.get("device", "auto")),
            "data_language": str(run.get("data_language", "bilingual")),
            "_dataset_path": str(dataset), "_resume_model_id": model_id,
            "_resume_checkpoint": str(checkpoint), "_resume_run_id": selected_id,
            **config,
        }
        text = dataset.read_text(encoding="utf-8")
        with self._state_lock:
            self._pending_training = None
            self._stop_event.clear()
            self._stop_reason = ""
            self._delete_after_stop = False
            self._training.update(status="preparing", message="正在从 checkpoint 继续训练")

        def worker() -> None:
            try:
                self._run_local_training(payload, text)
            except Exception as exc:
                with self._state_lock:
                    self._training.update(status="failed", message=str(exc))

        self._work_thread = threading.Thread(target=worker, name="orbit-continue-training", daemon=True)
        self._work_thread.start()
        return self.training_state()

    def hub_settings(self) -> dict[str, Any]:
        return self.hub.public_settings()

    def hub_login(self, payload: dict[str, Any], *, register: bool = False) -> dict[str, Any]:
        return self.hub.authenticate(str(payload.get("url", "")), str(payload.get("email", "")), str(payload.get("password", "")), register=register)

    def hub_logout(self) -> dict[str, Any]:
        return self.hub.logout()

    def hub_upload_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._hub_upload)

    def start_hub_upload(self, model_id: str) -> dict[str, Any]:
        checkpoint = self._checkpoint_for(model_id)
        metadata = self._model_metadata(model_id)
        with self._state_lock:
            if self._hub_upload.get("status") in {"hashing", "uploading"}:
                raise RuntimeError("已有模型正在上传")
            self._hub_upload = {"status": "hashing", "progress": 0, "message": "正在准备模型", "model": model_id}

        def progress(current: int, total: int, message: str) -> None:
            with self._state_lock:
                self._hub_upload.update(
                    status="uploading" if "上传" in message else "hashing",
                    progress=round(current / max(1, total) * 100, 1), message=message,
                )

        def worker() -> None:
            try:
                result = self.hub.upload_model(checkpoint, metadata, progress)
                with self._state_lock:
                    self._hub_upload.update(status="pending_review", progress=100, message="上传完成，等待管理员审核", result=result)
            except Exception as exc:
                with self._state_lock:
                    self._hub_upload.update(status="failed", message=str(exc))

        threading.Thread(target=worker, name="orbit-hub-upload", daemon=True).start()
        return self.hub_upload_state()

    def start_hub_job_upload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create the GPU bundle locally, then import that same bundle to Hub."""
        with self._state_lock:
            if self._hub_upload.get("status") in {"hashing", "uploading"}:
                raise RuntimeError("已有文件正在导入服务器")
        train_cfg = TrainingConfig(
            steps=int(payload.get("steps", 1000)),
            batch_size=int(payload.get("batch_size", 1)),
            seq_len=int(payload.get("seq_len", 2048)),
            grad_accum=int(payload.get("grad_accum", 8)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            warmup_steps=int(payload.get("warmup_steps", 100)),
            weight_decay=float(payload.get("weight_decay", 0.1)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
            precision=str(payload.get("precision", "auto")),
            scheduler=str(payload.get("scheduler", "cosine")),
            checkpoint_every=int(payload.get("checkpoint_every", 500)),
            seed=int(payload.get("seed", 42)),
        )
        train_cfg.validate()
        preset = str(payload.get("preset", "1b")).lower()
        text = str(payload.get("text", "")).strip() or ("Orbit training sample. " * 100)
        assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else None
        if assistant:
            assistant = dict(assistant)
            assistant.pop("api_key", None)
            assistant.pop("key", None)
        model_name = str(payload.get("model_name", "orbit"))
        bundle = create_job_bundle(
            self.jobs_root, preset, train_cfg.steps, train_cfg.batch_size,
            train_cfg.seq_len, train_cfg.learning_rate, text,
            training_config=train_cfg, model_name=model_name,
            data_language=str(payload.get("data_language", "bilingual")),
            assistant=assistant,
        )
        cfg = OrbitConfig.for_preset(preset)
        with self._state_lock:
            if self._hub_upload.get("status") in {"hashing", "uploading"}:
                raise RuntimeError("已有文件正在导入服务器")
            self._hub_upload = {
                "status": "hashing", "progress": 0,
                "message": "已在本机生成 GPU 训练包，准备导入服务器",
                "model": model_name, "kind": "gpu_training_bundle", "bundle": str(bundle),
            }

        def progress(current: int, total: int, message: str) -> None:
            with self._state_lock:
                self._hub_upload.update(
                    status="uploading" if "导入" in message or "上传" in message else "hashing",
                    progress=round(current / max(1, total) * 100, 1), message=message,
                )

        def worker() -> None:
            try:
                result = self.hub.upload_training_bundle(bundle, {
                    "name": f"Orbit GPU training · {model_name}",
                    "preset": preset,
                    "parameters": cfg.estimate_parameters(),
                    "description": (
                        "GPU training bundle generated locally by Orbit. "
                        "AI-assisted and human-authored training use the same bundle format; "
                        "the server does not execute it automatically."
                    ),
                }, progress)
                with self._state_lock:
                    self._hub_upload.update(
                        status="uploaded", progress=100,
                        message="训练包已导入服务器，等待管理员审核", result=result,
                    )
            except Exception as exc:
                with self._state_lock:
                    self._hub_upload.update(status="failed", message=str(exc))

        threading.Thread(target=worker, name="orbit-hub-job-upload", daemon=True).start()
        return self.hub_upload_state()

    def _remove_model_files(self, model_id: str) -> list[str]:
        removed: list[str] = []
        for path in (self.models_root / f"{model_id}.pt", self._metadata_path(model_id), self.models_root / f"{model_id}.gguf"):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
        return removed

    def delete_model(self, model_id: str, confirmation: str) -> dict[str, Any]:
        checkpoint = self._checkpoint_for(model_id)
        if confirmation != model_id:
            raise ValueError("删除确认必须与模型名称完全一致")
        with self._state_lock:
            if self._hub_upload.get("model") == model_id and self._hub_upload.get("status") in {"hashing", "uploading"}:
                raise RuntimeError("模型正在上传，暂时不能删除")
            loading_id = self._loading.get("model_id")
            load_thread = self._load_thread if loading_id == model_id else None
            loading_process = self._model if load_thread is not None else None
            if load_thread is not None and load_thread.is_alive():
                self._load_cancel.set()
                if loading_process is not None and loading_process.poll() is None:
                    loading_process.terminate()
        if load_thread is not None and load_thread is not threading.current_thread():
            load_thread.join(timeout=5)
            if load_thread.is_alive():
                raise RuntimeError("模型仍在加载，已请求停止；请稍后再删除")
        if self.active_model_id == model_id:
            self.unload_model()
        removed = self._remove_model_files(model_id)
        if not removed:
            raise FileNotFoundError(f"找不到可删除的本地模型：{model_id}")
        return {"status": "deleted", "model": model_id, "removed": removed, "training_history_preserved": True}

    def _delete_stopped_training_model(self) -> dict[str, Any]:
        """Delete the checkpoint belonging to a training job that already stopped.

        A safe stop deliberately leaves the checkpoint on disk.  The normal
        model-delete endpoint is useful once that checkpoint appears in the
        model list, but the training page also needs a direct way to remove an
        unfinished run—even when the worker stopped before writing its first
        checkpoint.  Keep the run record and its dataset so the training
        history remains inspectable.
        """
        with self._state_lock:
            state = dict(self._training)
        status = str(state.get("status", ""))
        model_id = str(state.get("model_id") or "").strip()
        run_id = str(state.get("run_id") or "").strip()
        if status not in {"stopped", "failed", "stopping", "waiting_memory", "needs_memory"}:
            raise RuntimeError("只能删除已经停止的训练模型")
        if not model_id:
            raise RuntimeError("这次训练没有生成可删除的模型文件")
        if Path(model_id).name != model_id or model_id in {"", ".", ".."}:
            raise ValueError("无效的模型名称")

        with self._state_lock:
            if self._hub_upload.get("model") == model_id and self._hub_upload.get("status") in {"hashing", "uploading"}:
                raise RuntimeError("模型正在上传，暂时不能删除")

        # A user may have loaded the checkpoint after stopping.  Reuse the
        # normal unload path before removing the exact model files.
        if self.active_model_id == model_id:
            self.unload_model()
        removed = self._remove_model_files(model_id)

        if run_id and Path(run_id).name == run_id:
            run_path = self.training_runs_root / run_id / "run.json"
            if run_path.is_file():
                try:
                    run = json.loads(run_path.read_text(encoding="utf-8"))
                    if isinstance(run, dict):
                        run.update(
                            status="stopped_deleted",
                            completed_at=run.get("completed_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            message="训练已停止，未完成模型已删除",
                        )
                        self._save_run(run)
                except (OSError, json.JSONDecodeError):
                    pass

        with self._state_lock:
            self._pending_training = None
            self._training.update(status="stopped_deleted", message="训练已停止，未完成模型已删除", checkpoint="")
        return {
            "status": "stopped_deleted",
            "model": model_id,
            "removed": removed,
            "training_history_preserved": True,
        }

    def stop_training(self, delete_checkpoint: bool = False) -> dict[str, Any]:
        with self._state_lock:
            running = bool(self._work_thread and self._work_thread.is_alive())
            current_status = str(self._training.get("status", ""))
        if delete_checkpoint and not running and current_status in {"stopped", "failed", "stopping", "waiting_memory", "needs_memory"}:
            return {**self.training_state(), **self._delete_stopped_training_model()}
        if not delete_checkpoint and not running and current_status == "stopping":
            return self.training_state()
        with self._state_lock:
            if not running:
                raise RuntimeError("当前没有正在运行的训练或数据生成任务")
            self._delete_after_stop = bool(delete_checkpoint)
            self._stop_reason = "用户已请求停止并删除未完成模型" if delete_checkpoint else "用户已请求安全停止"
            self._stop_event.set()
            self._training.update(
                status="stopping",
                message=("正在停止；训练完成前生成的模型将被删除" if delete_checkpoint else "正在安全停止；若已开始训练，将原子保存 checkpoint"),
            )
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
                if self._load_cancel.is_set():
                    raise RuntimeError("模型加载已取消")
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
            self._load_cancel.clear()
            self._set_loading("queued", 0, "等待加载", model_id)

        def worker() -> None:
            try:
                self.load_model(model_id)
            except Exception as exc:
                self.unload_model()
                if self._load_cancel.is_set():
                    self._set_loading("idle", 0, "模型加载已取消", None)
                else:
                    self._set_loading("failed", 0, str(exc), model_id)

        self._load_thread = threading.Thread(target=worker, name="orbit-model-loader", daemon=True)
        self._load_thread.start()
        return self.loading_state()

    @staticmethod
    def _release_memory() -> None:
        gc.collect()

    def unload_model(self) -> dict[str, Any]:
        # Set cancellation before taking the model lock. Loading holds this
        # lock while waiting for worker progress; this keeps the unload API
        # responsive and lets the worker exit cleanly instead of deadlocking.
        self._load_cancel.set()
        with self._model_lock:
            previous = self._model_id
            process = self._model
            released_bytes = 0
            if process is not None and process.poll() is None:
                try:
                    import psutil
                    released_bytes = psutil.Process(process.pid).memory_info().rss
                except (ImportError, psutil.Error):
                    pass
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
        return {
            "status": "unloaded" if previous else "already_unloaded",
            "previous_model": previous, "released_bytes": released_bytes,
            "memory_available_gb": resource_snapshot(self.data_root).get("memory_available_gb"),
        }

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
        if identity_challenge(prompt):
            resolved_model = model_id or self.active_model_id or (self.list_models()[0]["id"] if self.list_models() else "orbit")
            resolved_name = self._model_metadata(resolved_model).get("name", resolved_model) if resolved_model != "orbit" else "Orbit"
            return {
                "model": resolved_model,
                "model_name": resolved_name,
                "content": identity_response(prompt),
            }
        memory_context = self.memory.system_context()
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
            metadata = self._model_metadata(str(self._model_id))
            self._model.stdin.write(json.dumps({
                "command": "chat", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "system_prompt": ORBIT_IDENTITY,
                "memory_context": memory_context,
                "model_name": metadata.get("name", self._model_id),
            }, ensure_ascii=False) + "\n")
            self._model.stdin.flush()
            line = self._model.stdout.readline()
            if not line:
                raise RuntimeError("隔离推理进程意外退出")
            response = json.loads(line)
            if response.get("type") != "result":
                raise RuntimeError(str(response.get("error", "模型生成失败")))
            self._last_model_use = time.monotonic()
            metadata = self._model_metadata(str(self._model_id))
            return {"model": self._model_id, "model_name": metadata.get("name", self._model_id), "content": str(response.get("content", ""))}

    def export_model(self, model_id: str, target: str) -> dict[str, Any]:
        checkpoint = self._checkpoint_for(model_id)
        from .exports import create_model_export

        archive = create_model_export(
            project_root=Path(__file__).resolve().parents[1], data_root=self.data_root,
            model_id=model_id, checkpoint=checkpoint, metadata=self._model_metadata(model_id), target=target,
        )
        return {"model": model_id, "target": target, "filename": archive.name, "path": str(archive)}

    @property
    def active_model_id(self) -> str | None:
        return self._model_id if self._model is not None and self._model.poll() is None else None

    @property
    def active_model_name(self) -> str | None:
        model_id = self.active_model_id
        if not model_id:
            return None
        return str(self._model_metadata(model_id).get("name", model_id))
