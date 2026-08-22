from __future__ import annotations

import gc
import json
import os
import re
import select
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import OrbitConfig
from .community import CommunityStore
from .conversations import ConversationStore
from .hub import OrbitHubClient
from .jobs import create_job_bundle
from .memory import LongTermMemory
from .identity import (
    ORBIT_SYSTEM_PROMPT,
    ORBIT_TRAINING_ANCHOR,
)
from .settings import OrbitSettings
from .resources import TRAINING_MEMORY_RESERVE_GB, memory_is_critical, require_checkpoint_load_capacity, require_training_capacity, resource_snapshot
from .teacher import TeacherConfig, generate_dataset
from .training_config import TrainingConfig


PRESETS = ("300m", "1b", "3b", "7b", "14b", "38b")
MAX_MODEL_DOWNLOAD_BYTES = 64 * 1024 * 1024 * 1024
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
        self.training_state_path = self.data_root / "training-state.json"
        self._teacher_settings = self._load_teacher_settings()
        self.community = CommunityStore(self.data_root)
        self.conversations = ConversationStore(self.data_root)
        self.memory = LongTermMemory(self.data_root)
        self.hub = OrbitHubClient(self.data_root)
        self.settings = OrbitSettings(self.data_root)
        self._hub_upload: dict[str, Any] = {"status": "idle", "progress": 0, "message": "尚未上传", "model": None}
        self._model_download: dict[str, Any] = {"status": "idle", "progress": 0, "message": "没有正在下载模型", "model": None}
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
            "assisted": False, "generated_content_available": False,
            "generated_content_path": "", "generated_content_bytes": 0,
        }
        self._restore_training_state()
        self._loading: dict[str, Any] = {
            "status": "idle", "progress": 0, "message": "没有正在加载的模型", "model_id": None,
        }
        threading.Thread(target=self._idle_janitor, name="orbit-model-janitor", daemon=True).start()

    def _restore_training_state(self) -> None:
        """Restore a resumable run after the local API process restarts.

        The training form is intentionally reset on re-entry, but the run
        lifecycle must remain visible so the user can continue or delete a
        stopped run without pretending that the service is still training.
        Live worker processes are not restored as running; only durable,
        non-running states are safe to expose here.
        """
        restorable = {"stopped", "failed", "waiting_memory", "needs_memory"}
        # AI corpus generation can fail before a normal training run exists.
        # Restore its durable partial-corpus snapshot first so closing/reopening
        # Orbit never makes completed teacher batches disappear.
        if self.training_state_path.is_file():
            try:
                saved = json.loads(self.training_state_path.read_text(encoding="utf-8"))
                generated = Path(str(saved.get("generated_content_path") or ""))
                if (
                    isinstance(saved, dict)
                    and str(saved.get("status", "")) in restorable | {"generating"}
                    and generated.is_file()
                    and generated.resolve().parent == self.datasets_root.resolve()
                ):
                    if saved.get("status") == "generating":
                        saved["status"] = "failed"
                        saved["message"] = "Orbit 重新启动，AI 生成已中断；已完成的样本仍保存在本机，可查看或复制"
                    saved["generated_content_available"] = generated.stat().st_size > 0
                    saved["generated_content_bytes"] = generated.stat().st_size
                    self._training.update(saved)
                    return
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        candidates = []
        for path in self.training_runs_root.glob("*/run.json"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
        for _, path in sorted(candidates, reverse=True):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict) or str(row.get("status", "")) not in restorable:
                continue
            model_id = str(row.get("model_id") or "").strip() or None
            safe_model_id = Path(model_id).name if model_id else ""
            checkpoint = str(self.models_root / f"{safe_model_id}.pt") if safe_model_id == model_id else ""
            self._training.update(
                status=str(row.get("status")), step=int(row.get("step", 0) or 0),
                steps=int(row.get("steps", 0) or 0), loss=row.get("loss"),
                message=str(row.get("message") or "这次训练已停止，可以继续或删除"),
                model_id=model_id, model_name=str(row.get("model_name") or model_id or ""),
                checkpoint=checkpoint, device=str(row.get("device") or "auto"),
                phase="paused", dataset=str(row.get("dataset") or ""),
                run_id=str(row.get("id") or path.parent.name),
            )
            return

    def _save_generation_state(self) -> None:
        """Atomically persist non-secret AI generation progress."""
        with self._state_lock:
            state = {
                key: self._training.get(key)
                for key in (
                    "status", "phase", "step", "steps", "message", "model_id",
                    "teacher_model", "dataset", "assisted",
                    "generated_content_available", "generated_content_path",
                    "generated_content_bytes", "training_mode", "training_round",
                )
            }
        temporary = self.training_state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.training_state_path)

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
        examples = max(1, min(50_000, int(payload.get("examples", 20))))
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
        # On Apple Silicon, ``auto`` resolves to MPS in the bundled runtime.
        # The old recommendation left the 300M preset at 1024 tokens and eight
        # accumulation passes, which is a poor local-MPS starting point even
        # when memory is available.  Keep the first run bounded so it can
        # finish in a useful time; users can still edit these fields upward.
        local_mps = device == "mps" or (device == "auto" and sys.platform == "darwin")
        if local_mps:
            seq_len = min(seq_len, 512)
        scale_examples = {"300m": 2_000, "1b": 5_000, "3b": 10_000, "7b": 20_000, "14b": 30_000, "38b": 50_000}[preset]
        goal_chars = max(0, min(20_000, int(payload.get("goal_chars", 0))))
        recommended_examples = min(50_000, scale_examples + min(10_000, goal_chars // 100))
        # A freshly reset form has not been edited by the user yet. In that
        # state the sample count and the advanced parameters must be one
        # coherent recommendation: do not calculate 100 steps from the
        # temporary 20-sample placeholder and then replace only the sample
        # field with 2,000. Once the user edits the sample count, the browser
        # sends use_recommended_examples=false and their exact value wins.
        use_recommended_examples = bool(payload.get("use_recommended_examples", False))
        if use_recommended_examples:
            data_units = recommended_examples
        elif bool(payload.get("assisted")):
            data_units = examples
        else:
            data_units = max(1, text_chars // max(256, seq_len))
        steps = max(100, min(2000, data_units * (20 if bool(payload.get("assisted")) else 8)))
        optimization_goal = str(payload.get("optimization_goal", "balanced")).strip().lower() or "balanced"
        if optimization_goal not in {"fast", "memory", "quality", "balanced"}:
            raise ValueError("训练优化目标必须是省时间、省内存、效果优先或平衡")
        if optimization_goal == "fast":
            seq_len = min(seq_len, 256)
            steps = max(50, min(2000, round(steps * 0.4)))
            recommended_examples = min(50_000, max(50, round(recommended_examples * 0.7)))
        elif optimization_goal == "memory":
            seq_len = min(seq_len, 256)
            steps = max(50, min(2000, round(steps * 0.8)))
            recommended_examples = min(50_000, max(50, round(recommended_examples * 0.8)))
        elif optimization_goal == "quality":
            seq_len = max(seq_len, min(base.seq_len, 1024))
            steps = max(100, min(2000, round(steps * 1.5)))
            recommended_examples = min(50_000, max(recommended_examples, round(recommended_examples * 1.25)))
        warmup = max(10, min(200, steps // 10))
        checkpoint_every = max(25, min(250, steps // 5))
        config = base.with_overrides(
            steps=steps,
            batch_size=1 if pressure > 0.55 or device == "cpu" or local_mps else base.batch_size,
            seq_len=seq_len,
            grad_accum=(1 if optimization_goal in {"fast", "memory"} or preset == "300m" else max(1, base.grad_accum // 4)) if local_mps else (1 if optimization_goal == "memory" else base.grad_accum),
            warmup_steps=warmup,
            checkpoint_every=checkpoint_every,
            precision="fp32" if local_mps else "auto",
        )
        if optimization_goal == "memory":
            config = config.with_overrides(batch_size=1, seq_len=min(config.seq_len, 256), grad_accum=1)
        elif optimization_goal == "fast":
            config = config.with_overrides(batch_size=1, grad_accum=1, checkpoint_every=max(steps, 1000))
        elif optimization_goal == "quality":
            config = config.with_overrides(steps=min(2000, max(config.steps, round(config.steps * 1.2))))
        # Manual literature is measured in UTF-8 characters, while AI
        # assistance is measured in generated examples. Keep both linked to
        # the selected optimization target so the UI can recommend them
        # together. This is deliberately conservative for mixed-language text.
        recommended_manual_chars = recommended_examples * 600
        recommended_manual_words = max(1, round(recommended_manual_chars / 1.6))
        base_model = str(payload.get("base_model", "")).strip()
        requested_mode = str(payload.get("training_mode", "")).strip().lower()
        base_source = str(payload.get("base_model_source", "local")).strip().lower() or "local"
        if requested_mode not in {"pretraining", "fine_tuning"}:
            requested_mode = "fine_tuning" if base_model else "pretraining"
        training_round = self._training_round(payload, base_model or None)
        parent_preset = None
        if requested_mode == "fine_tuning" and base_model:
            parent_preset = str(self._model_metadata(base_model).get("preset", "")).lower() or None
        # Fine-tuning should make smaller, conservative updates to an existing
        # checkpoint. There is no universal magic value, but a lower learning
        # rate and a bounded first pass are safer defaults than reusing the
        # from-scratch pretraining schedule.
        if requested_mode == "fine_tuning":
            config = config.with_overrides(
                learning_rate=min(config.learning_rate, 1e-4),
                steps=max(60, min(config.steps, 600)),
            )
        mode_valid = (
            bool(base_model)
            if requested_mode == "fine_tuning" or training_round > 1
            else not base_model
        )
        mode_valid = mode_valid and not (requested_mode == "fine_tuning" and parent_preset in PRESETS and parent_preset != preset)
        if requested_mode == "pretraining" and base_source != "local":
            mode_valid = False
        requested_steps = max(1, int(payload.get("steps", 0) or config.steps))
        requested_batch = max(1, int(payload.get("batch_size", 0) or config.batch_size))
        requested_accum = max(1, int(payload.get("grad_accum", 0) or config.grad_accum))
        requested_seq = max(8, int(payload.get("seq_len", 0) or config.seq_len))
        requested_examples = max(1, min(50_000, int(payload.get("examples", 20))))
        text_bytes = max(0, min(500_000_000, int(payload.get("text_bytes", 0) or 0)))
        estimated_sequences = text_bytes // max(1, requested_seq + 1) if text_bytes else 0
        effective_batch = requested_batch * requested_accum
        estimated_updates_per_corpus = max(1, (estimated_sequences + effective_batch - 1) // effective_batch) if estimated_sequences else 0
        advice: list[dict[str, str]] = []
        if requested_mode == "fine_tuning" and parent_preset in PRESETS and parent_preset != preset:
            advice.append({
                "severity": "warning", "code": "finetune_scale_locked",
                "zh": f"微调基础模型是 {parent_preset.upper()}，但当前选择了 {preset.upper()}。微调不会改变模型规模；请把模型规模改回 {parent_preset.upper()}，只调整训练参数。",
                "en": f"The fine-tuning base model is {parent_preset.upper()}, but the selected scale is {preset.upper()}. Fine-tuning never changes model size; select {parent_preset.upper()} and adjust training parameters only.",
            })
        goal_titles = {
            "fast": {"zh": "省时间", "en": "Time saving"},
            "memory": {"zh": "省内存", "en": "Memory saving"},
            "quality": {"zh": "效果优先（不计时间）", "en": "Quality first (time is not a constraint)"},
            "balanced": {"zh": "平衡", "en": "Balanced"},
        }
        advice.append({
            "severity": "info", "code": "optimization_goal",
            "zh": f"当前参数目标：{goal_titles[optimization_goal]['zh']}。样本数、步数、序列长度、梯度累计和 checkpoint 频率都会随目标调整；仍可在高级参数中手动修改。",
            "en": f"Optimization target: {goal_titles[optimization_goal]['en']}. Sample count, steps, sequence length, accumulation and checkpoint frequency are adjusted together; you can still edit advanced values manually.",
        })
        policy = self._corpus_policy(requested_mode)
        advice.append({
            "severity": "info", "code": "corpus_policy",
            "zh": f"语料建议：{policy['description']}。人工内容负责你希望 Orbit 记住的专属知识、代码规范、产品资料和目标；AI 辅助负责基础认知、通用语言能力和基础任务。结构化任务包括分类/情感分析、NER、代码/SQL、摘要和扩写/润色；这些是单轮输入→输出，不是聊天气泡。",
            "en": f"Corpus policy: {policy['description']}. Your manual corpus teaches Orbit the private knowledge, coding conventions, product material and goals you want it to remember; AI assistance supplies basic knowledge, general language ability and foundational tasks. Structured tasks include classification/sentiment, NER, code/SQL, summarization and expansion/polishing; these are single-turn input→output tasks, not chat bubbles.",
        })
        advice.append({
            "severity": "info", "code": "manual_ai_roles",
            "zh": "如果同时使用人工语料和 AI 辅助：AI 生成基础认知、通用语言/代码模式和少量基础任务；人工语料只放你希望 Orbit 学会的专属内容。两部分会合并训练，AI 不会覆盖人工内容。",
            "en": "When both sources are enabled: AI generates foundational knowledge, general language/code patterns and basic tasks; your manual corpus contains the specific material you want Orbit to learn. Both are merged into one run, and AI does not replace your manual content.",
        })
        advice.append({
            "severity": "info", "code": "research_recipe",
            "zh": "研究型建议：预训练应按 token 总量规划，而不是只重复少量样本；先做去重、语言/质量过滤、代码和文献混合，再保留验证集。微调从较低学习率和约 1～3 个数据遍历开始，验证集变好再增加步数，避免小数据过拟合。",
            "en": "Research-informed recipe: plan pretraining by total tokens instead of repeatedly cycling a tiny sample set; deduplicate and quality-filter before mixing documents and code, and keep a held-out validation set. For fine-tuning, start with a lower learning rate and roughly 1–3 dataset passes, then add steps only when held-out validation improves.",
        })
        if requested_mode == "fine_tuning":
            mode_title = {"zh": "微调（专业文献 + 验证驱动的任务/对话混合）", "en": "Fine-tuning (specialized documents + validation-driven task/dialogue mix)"}
            if not base_model:
                advice.append({
                    "severity": "warning", "code": "finetune_parent_required",
                    "zh": "你选择了微调，但还没有选择已有模型。请先加载一个本地 checkpoint；否则请选择‘预训练（从零开始）’。",
                    "en": "Fine-tuning is selected, but no existing model is selected. Choose a local checkpoint, or switch to ‘Pretraining (from scratch)’.",
                })
            if requested_examples < 100 and requested_steps >= max(200, requested_examples * 4):
                advice.append({
                    "severity": "warning", "code": "finetune_overfit",
                    "zh": f"当前是微调：{requested_examples} 个样本配 {requested_steps} 步，可能过拟合。样本可以增加，但必须先去重、质检并保留验证集；只有验证集继续改善时才增加步数。",
                    "en": f"This is fine-tuning: {requested_examples} samples with {requested_steps} steps may overfit. More data is useful only after deduplication and quality checks; increase steps only while held-out validation improves.",
                })
            advice.append({
                "severity": "info", "code": "finetune_data",
                "zh": "微调不采用固定百分比：先保留基础知识，再根据领域、结构化任务、代码/数学、对话和身份的验证集表现动态补样；优先增加高质量、去重后且能改善验证集的类别。",
                "en": "Fine-tuning uses no fixed percentages: preserve general knowledge, then adapt domain, structured-task, code/math, dialogue and identity data according to held-out validation; prioritize high-quality, deduplicated data that improves validation.",
            })
        else:
            mode_title = (
                {"zh": "预训练第 1 次（文献为主 + 少量对话）", "en": "Pretraining round 1 (document-first + a small amount of dialogue)"}
                if training_round == 1
                else {"zh": f"Orbit 继续预训练第 {training_round} 次（文献为主 + 少量对话）", "en": f"Orbit continued pretraining round {training_round} (document-first + a small amount of dialogue)"}
            )
            if training_round > 1 and not base_model:
                advice.append({
                    "severity": "warning", "code": "continued_pretrain_parent_required",
                    "zh": f"这是 Orbit 继续预训练第 {training_round} 次，必须选择第 {training_round - 1} 次生成的父模型；否则会重新从随机权重开始。",
                    "en": f"This is Orbit continued pretraining round {training_round}; select the model from round {training_round - 1} or it will start from random weights again.",
                })
            if training_round == 1 and base_model:
                advice.append({
                    "severity": "warning", "code": "first_pretrain_no_parent",
                    "zh": "预训练第 1 次从随机权重开始，不能选择父模型；如果要接着 Orbit 训练，请把训练次数改为父模型次数 + 1。",
                    "en": "Pretraining round 1 starts from random weights and cannot use a parent; to continue Orbit training, set the round to the parent round + 1.",
                })
            if base_source != "local":
                advice.append({
                    "severity": "warning", "code": "external_pretrain_forbidden",
                    "zh": "外部或下载模型不能从零训练；它必须作为基础模型进入微调。请选择‘微调（已有模型）’，并从模型列表选择 checkpoint。",
                    "en": "External or downloaded models cannot be trained from scratch. Use them as a fine-tuning base: choose ‘Fine-tuning (existing model)’ and select a checkpoint.",
                })
            if requested_examples < 500 or requested_steps < 1000:
                advice.append({
                    "severity": "warning", "code": "pretrain_small",
                    "zh": f"当前是从零预训练实验，不是微调：{requested_examples} 个 AI 样本和 {requested_steps} 步不足以训练出可靠的通用聊天机器人。建议使用更多样、更大规模的文献语料，并按 token 总量规划训练。",
                    "en": f"This is a from-scratch pretraining experiment, not fine-tuning: {requested_examples} AI samples and {requested_steps} steps are not enough for a reliable general chatbot. Use a much larger, more diverse corpus and plan by total tokens.",
                })
            advice.append({
                "severity": "info", "code": "pretrain_units",
                "zh": "预训练第 1～N 次都以多篇文献、教材、技术资料或代码为主，只加入少量对话和身份样本；‘步数 × batch × 梯度累计’不等于数据集遍历次数。",
                "en": "Pretraining rounds 1–N are document-first: use many papers, books, technical references or code, with only a small dialogue and identity tail. Steps × batch × accumulation is not the number of corpus passes.",
            })
        if requested_seq >= 2048:
            advice.append({
                "severity": "warning", "code": "long_context",
                "zh": f"序列长度为 {requested_seq}，会明显增加内存和每步时间；如果目标是先验证流程，建议从 512–1024 开始。",
                "en": f"Sequence length is {requested_seq}, which substantially increases memory and step time. For a first validation run, start around 512–1,024.",
            })
        if requested_batch * requested_accum >= 8:
            advice.append({
                "severity": "info", "code": "effective_batch",
                "zh": f"有效批次约为 {requested_batch * requested_accum}；它主要提高梯度稳定性，但会延长每次参数更新所需时间。",
                "en": f"The effective batch is about {requested_batch * requested_accum}; it mainly stabilizes gradients but lengthens each optimizer update.",
            })
        training_advice = {
            "mode": requested_mode,
            "mode_title": mode_title,
            "base_model": base_model or None,
            "base_model_source": base_source,
            "base_model_preset": parent_preset,
            "model_scale_locked": bool(requested_mode == "fine_tuning" and parent_preset in PRESETS),
            "training_round": training_round,
            "corpus_policy": self._corpus_policy(requested_mode),
            "optimization_goal": optimization_goal,
            "optimization_goal_title": goal_titles[optimization_goal],
            "mode_valid": mode_valid,
            "items": advice,
            "requested": {
                "steps": requested_steps, "examples": requested_examples, "training_round": training_round,
                "batch_size": requested_batch, "grad_accum": requested_accum, "seq_len": requested_seq,
            },
            "dataset_estimate": {
                "text_bytes": text_bytes,
                "sequence_samples": estimated_sequences,
                "effective_batch": effective_batch,
                "updates_per_corpus": estimated_updates_per_corpus,
                "note": "根据 UTF-8 字节数和序列长度估算；当前训练使用随机窗口，实际覆盖率不是固定遍历次数。",
            },
        }
        # Pre-training estimates are deliberately labeled as rough. Once a
        # run starts, training_state() replaces them with measured ETA from
        # completed optimizer steps. The baseline is calibrated to the local
        # 300M MPS path and scaled by parameter count and token work.
        parameter_scale = max(0.25, (model.estimate_parameters() / 308_450_304) ** 0.85)
        # Attention work grows faster than linearly with context length.  This
        # is still a rough pre-run estimate; live ETA is always based on real
        # completed optimizer steps.
        token_work = (config.seq_len / 512) ** 1.5 * config.batch_size * (config.grad_accum / 8)
        device_factor = {"cpu": 2.0, "mps": 0.65, "cuda": 0.45}.get("mps" if local_mps else device, 1.0)
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
            "recommended_manual_chars": recommended_manual_chars,
            "recommended_manual_words": recommended_manual_words,
            "recommended_manual_text": f"约 {recommended_manual_chars:,} 个字符（约 {recommended_manual_words:,} 个词；中文可按字符数准备）",
            "manual_content_recommendation": self._corpus_policy(requested_mode)["manual_recommendation"],
            "ai_content_recommendation": self._corpus_policy(requested_mode)["ai_recommendation"],
            "mixed_content_strategy": self._corpus_policy(requested_mode)["mixed_strategy"],
            "config": config.__dict__,
            "estimated_step_seconds": round(estimated_step_seconds),
            "estimated_training_seconds": estimated_training_seconds,
            "estimated_peak_memory_gb": round(estimated_peak_memory, 1),
            "estimated_activation_delta_gb": round(max(0.0, estimated_peak_memory - required), 1),
            "estimate_note": "粗略估算；训练开始后会用实际步速和 ETA 替换。教师 API 生成时间另计。",
            "training_advice": training_advice,
            "training_mode": requested_mode,
            "training_round": training_round,
            "base_model_source": base_source,
            "optimization_goal": optimization_goal,
            "mode_valid": mode_valid,
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

    def generated_training_content(self) -> dict[str, Any]:
        """Return the current teacher-generated corpus for the local UI.

        The full corpus is deliberately kept out of ``/api/training`` and
        ``/api/system`` polling responses.  This endpoint is local-only like
        the rest of the UI and only reads a file owned by Orbit's datasets
        directory, so a user can inspect/copy the exact material before or
        while the isolated worker trains on it.
        """
        with self._state_lock:
            state = dict(self._training)
        if not state.get("assisted") and not state.get("generated_content_available"):
            return {"available": False, "content": "", "bytes": 0}
        raw_path = str(state.get("generated_content_path") or state.get("dataset") or "").strip()
        if not raw_path:
            return {"available": False, "content": "", "bytes": 0}
        candidate = Path(raw_path)
        try:
            if candidate.resolve().parent != self.datasets_root.resolve() or not candidate.is_file():
                return {"available": False, "content": "", "bytes": 0}
            content = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return {"available": False, "content": "", "bytes": 0}
        return {
            "available": bool(content),
            "content": content,
            "bytes": len(content.encode("utf-8")),
            "assisted": True,
            "model_id": state.get("model_id"),
            "run_id": state.get("run_id"),
            "training_mode": state.get("training_mode"),
            "training_round": state.get("training_round"),
        }

    def loading_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._loading)

    def model_download_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._model_download)

    @staticmethod
    def _download_url(value: str) -> str:
        url = str(value or "").strip()
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("模型下载地址必须是不带账号密码的 HTTPS 地址")
        host = parsed.hostname.lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            raise ValueError("模型下载地址不能指向本机或局域网地址")
        try:
            import ipaddress
            address = ipaddress.ip_address(host)
            if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
                raise ValueError("模型下载地址不能指向本机或私有网络地址")
        except ValueError as exc:
            if "不能指向" in str(exc):
                raise
            # Domain names are allowed; urllib will perform the HTTPS request.
        return url

    @staticmethod
    def _downloaded_checkpoint_config(path: Path) -> tuple[OrbitConfig, str]:
        """Validate a downloaded file without placing arbitrary weights in the model list."""
        try:
            import torch
            try:
                payload = torch.load(path, map_location="meta", weights_only=True)
            except TypeError:  # pragma: no cover - compatibility with older torch
                payload = torch.load(path, map_location="meta")
        except Exception as exc:
            raise ValueError("下载文件不是可读取的 Orbit checkpoint") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict) or not isinstance(payload.get("model"), dict):
            raise ValueError("下载文件不是 Orbit checkpoint，不能作为微调基础模型")
        try:
            config = OrbitConfig(**payload["config"])
            config.validate()
        except Exception as exc:
            raise ValueError("下载模型的架构配置不是 Orbit 格式") from exc
        if config.vocab_size != 256:
            raise ValueError("该模型使用其他 tokenizer/词表；当前 Orbit 训练器暂不支持 Qwen 等外部权重")
        for preset in PRESETS:
            if config == OrbitConfig.for_preset(preset):
                return config, preset
        if config == OrbitConfig.tiny():
            return config, "local"
        raise ValueError("该 Orbit checkpoint 的架构暂不在可微调的 300M–38B 档位中")

    def start_model_download(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._download_url(str(payload.get("url", "")))
        raw_name = str(payload.get("model_name", "")).strip()
        parsed = urlparse(url)
        fallback = Path(parsed.path).stem or "downloaded-orbit"
        model_id = self._safe_model_name(raw_name, fallback)
        checkpoint = self.models_root / f"{model_id}.pt"
        if checkpoint.exists():
            raise ValueError(f"模型名称已存在：{model_id}，请换一个名称")
        with self._state_lock:
            if self._model_download.get("status") in {"downloading", "validating"}:
                raise RuntimeError("已有模型正在下载")
            if self._work_thread is not None and self._work_thread.is_alive():
                raise RuntimeError("训练进行中，不能同时下载基础模型")
            self._model_download = {"status": "downloading", "progress": 0, "message": "正在下载模型", "model": model_id, "url": url}

        temporary = checkpoint.with_suffix(".pt.download")

        def worker() -> None:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Orbit/0.6 model downloader"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    total = int(response.headers.get("Content-Length", "0") or 0)
                    if total > MAX_MODEL_DOWNLOAD_BYTES:
                        raise ValueError("模型下载超过 Orbit 的 64GB 安全上限")
                    received = 0
                    with temporary.open("wb") as output:
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            received += len(block)
                            if received > MAX_MODEL_DOWNLOAD_BYTES:
                                raise ValueError("模型下载超过 Orbit 的 64GB 安全上限")
                            output.write(block)
                            with self._state_lock:
                                self._model_download.update(progress=round(received / total * 100, 1) if total else 0, received_bytes=received, total_bytes=total)
                with self._state_lock:
                    self._model_download.update(status="validating", message="正在验证 Orbit checkpoint", progress=100)
                config, preset = self._downloaded_checkpoint_config(temporary)
                metadata = {
                    "name": model_id, "display_name": model_id, "model_id": model_id,
                    "preset": preset, "parameters": config.estimate_parameters(),
                    "identity": "Orbit", "developer": "YUNSH", "system_prompt": ORBIT_IDENTITY,
                    "training_mode": "pretraining", "parent_model": None, "training_runs": [],
                    "identity_training_examples": ORBIT_TRAINING_ANCHOR,
                    "architecture": "orbit-hybrid-moe-v1", "ollama_ready": False,
                    "origin": "downloaded", "source_url": url, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                os.replace(temporary, checkpoint)
                self._write_model_metadata(model_id, metadata)
                with self._state_lock:
                    self._model_download.update(status="completed", message="模型已下载并加入基础模型列表", model=model_id, preset=preset)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                with self._state_lock:
                    self._model_download.update(status="failed", message=str(exc), error=str(exc))

        threading.Thread(target=worker, name="orbit-model-download", daemon=True).start()
        return self.model_download_state()

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
        # A JSON file is only metadata for a checkpoint.  Older interrupted
        # runs could leave that metadata behind after the checkpoint was never
        # created; it must not make a model appear to exist or reserve its
        # name forever.
        duplicate = (self.models_root / f"{normalized}.pt").is_file()
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
        generated_dataset = Path(str(row.get("generated_content_path", "")))
        row["generated_content"] = generated_dataset.read_text(encoding="utf-8") if generated_dataset.is_file() else ""
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

    @staticmethod
    def _training_mode(payload: dict[str, Any], *, require_valid: bool = True) -> tuple[str, str | None]:
        """Return the explicit training mode and its optional parent model.

        Older clients did not send ``training_mode``; infer it from the
        parent checkpoint for backwards compatibility.  New clients must not
        silently turn a requested fine-tune into random-init pretraining.
        """
        parent = str(payload.get("base_model", "")).strip() or None
        mode = str(payload.get("training_mode", "")).strip().lower()
        base_source = str(payload.get("base_model_source", "local")).strip().lower() or "local"
        if base_source not in {"local", "download", "external"}:
            raise ValueError("基础模型来源无效")
        if mode not in {"pretraining", "fine_tuning"}:
            mode = "fine_tuning" if parent else "pretraining"
        training_round = OrbitRuntime._training_round(payload, parent)
        if mode == "pretraining":
            if base_source != "local" and require_valid:
                raise ValueError("外部或下载模型不能从零预训练；请选择微调并选择一个基础模型")
            if training_round == 1:
                if parent and require_valid:
                    raise ValueError("预训练第 1 次不能选择父模型；如果要继续 Orbit 训练，请把训练次数设为父模型次数 + 1")
                parent = None
            elif require_valid and not parent:
                raise ValueError(f"预训练第 {training_round} 次必须选择第 {training_round - 1} 次的父模型")
        elif require_valid and not parent:
            raise ValueError("已选择微调，请先选择一个已有模型；或者切换为预训练（从零开始）")
        return mode, parent

    @staticmethod
    def _training_round(payload: dict[str, Any], parent_model: str | None = None) -> int:
        raw = payload.get("training_round")
        if raw is None or str(raw).strip() == "":
            return 2 if parent_model else 1
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("训练次数必须是大于等于 1 的整数") from exc
        if value < 1 or value > 1000:
            raise ValueError("训练次数必须在 1 到 1000 之间")
        return value

    @staticmethod
    def _corpus_policy(training_mode: str) -> dict[str, Any]:
        if training_mode == "fine_tuning":
            return {
                "name": "adaptive_quality_diverse_instruction_mix",
                "document_ratio": None,
                "structured_task_ratio": None,
                "dialogue_ratio": None,
                "initial_target_ranges": {
                    "general_knowledge": "15–25%",
                    "domain_documents": "20–35%",
                    "structured_tasks": "20–30%",
                    "code_math": "10–20%",
                    "dialogue_style": "5–15%",
                    "identity_safety": "2–5%",
                },
                "task_types": ["分类/情感分析", "实体抽取（NER）", "代码/SQL 生成", "摘要总结", "文本扩写/润色"],
                "description": "研究驱动的动态混合：保留基础知识，再根据领域、结构化任务、代码/数学、对话和身份的验证集表现自动补样，不固定为某个百分比",
                "manual_recommendation": "人工：专业文献、产品/领域知识、术语、代码规范、标注规则和希望模型遵守的回答格式；人工内容应以你真实希望它掌握的专属知识为主。",
                "ai_recommendation": "AI 辅助：生成基础认知、通用编程模式、结构化任务和少量基础对话，帮助模型学会输入→输出和指令遵循。",
                "mixed_strategy": "同时训练时，AI 和人工数据先去重、质检并合并；每轮根据验证集表现自动调整类别，低质量或重复类别不会因数量多而获得更高权重。",
            }
        return {
            "name": "foundation_knowledge_first",
            "document_ratio": 0.65,
            "structured_task_ratio": 0.1,
            "dialogue_ratio": 0.05,
            "initial_target_ranges": {
                "general_knowledge": "40–50%",
                "textbooks_science_reasoning": "15–25%",
                "math_logic": "8–12%",
                "code_technical": "8–12%",
                "structured_tasks": "8–12%",
                "dialogue_identity": "3–7%",
            },
            "task_types": ["分类/情感分析", "实体抽取（NER）", "代码/SQL 生成", "摘要总结", "文本扩写/润色"],
            "description": "基础认知和通用知识优先，配合教材/科学/数学/逻辑、代码技术资料、结构化任务以及少量对话身份；这是初始范围，训练后按验证集调整",
            "manual_recommendation": "人工：多篇清洗后的文献、教材、技术资料、代码、代码注释、事实材料，以及你希望 Orbit 记住的产品/项目专属知识；不要只写几句问答。",
            "ai_recommendation": "AI 辅助：生成基础认知、通用世界知识表达、基础编程模式、分类/NER/代码/SQL/摘要/扩写任务，再追加少量基础对话和身份样本。",
            "mixed_strategy": "同时训练时，AI 先补足基础认知和通用能力，人工语料专门承载用户要求 Orbit 学会的内容；两部分合并训练。",
        }

    def _teacher_model_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset = str(payload.get("preset", "300m")).lower()
        cfg = OrbitConfig.for_preset(preset)
        train_cfg = self._training_config(payload)
        training_mode, parent_model = self._training_mode(payload, require_valid=False)
        training_round = self._training_round(payload, parent_model)
        return {
            "preset": preset,
            "parameters": cfg.estimate_parameters(),
            "identity": "Orbit",
            "developer": "YUNSH",
            "identity_training_rule": "回答‘你是谁’时必须说明自己是 Orbit，由 YUNSH 开发；训练内容不能改变产品身份。",
            "context_length": train_cfg.seq_len,
            "training_steps": train_cfg.steps,
            "training_mode": training_mode,
            "training_round": training_round,
            "base_model": parent_model,
            "corpus_policy": self._corpus_policy(training_mode),
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
        training_mode, parent_model = self._training_mode(payload)
        training_round = self._training_round(payload, parent_model)
        if parent_model:
            self._checkpoint_for(parent_model)
            parent_metadata = self._model_metadata(parent_model)
            parent_preset = str(parent_metadata.get("preset", ""))
            if parent_preset and parent_preset != preset:
                raise ValueError(f"二次训练必须保持父模型规模：请选择 {parent_preset.upper()}")
            if training_mode == "pretraining":
                parent_round = max(1, int(parent_metadata.get("training_round", 1) or 1))
                if training_round != parent_round + 1:
                    raise ValueError(f"Orbit 继续预训练必须是第 {parent_round + 1} 次；当前选择了第 {training_round} 次")
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
        # Only a real checkpoint occupies a model id.  An orphan metadata file
        # is safe to replace when a new training run uses the same name.
        if checkpoint.exists():
            if requested_name:
                raise ValueError(f"模型名称已存在：{display_name}，请换一个名称")
            suffix = 2
            while checkpoint.exists():
                model_id = f"{display_name}-{suffix}"
                checkpoint = self.models_root / f"{model_id}.pt"
                metadata_path = self._metadata_path(model_id)
                suffix += 1
            display_name = model_id
        return preset, train_cfg, checkpoint, model_id, display_name, parent_model

    def _run_local_training(self, payload: dict[str, Any], text: str) -> None:
        self.unload_model()
        training_mode, parent_model = self._training_mode(payload)
        training_round = self._training_round(payload, parent_model)
        preset, train_cfg, checkpoint, model_id, display_name, parent_model = self._prepare_training(payload, text)
        with self._state_lock:
            # Publish the target before creating the worker.  This makes a
            # stop-and-delete request during the preparation window address
            # the exact model path even if no checkpoint has been written yet.
            self._training.update(model_id=model_id, model_name=display_name, checkpoint=str(checkpoint))
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
            "generated_content_path": str(payload.get("_generated_dataset_path") or dataset) if payload.get("_assisted") else "",
            "training_mode": training_mode,
            "training_round": training_round,
            "base_model_source": str(payload.get("base_model_source", "local")),
            "corpus_mode": "fine_tuning" if training_mode == "fine_tuning" else "pretraining",
            "corpus_plan": str(payload.get("corpus_plan", "")),
            "corpus_policy": self._corpus_policy(training_mode),
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
            "identity": "Orbit", "developer": "YUNSH", "system_prompt": ORBIT_IDENTITY,
            "training_mode": training_mode, "training_round": training_round, "parent_model": parent_model,
            "base_model_source": str(payload.get("base_model_source", "local")),
            "corpus_plan": str(payload.get("corpus_plan", "")),
            "corpus_policy": self._corpus_policy(training_mode),
            "identity_training_examples": ORBIT_TRAINING_ANCHOR,
            "quality_status": "unverified",
            "quality_note": "训练流程已完成，但模型回答质量尚未通过独立对话验证。",
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
            final_message = (self._stop_reason or "训练已停止，已原子保存当前 checkpoint") if stopped else "训练流程完成，模型已保存；对话质量尚未通过独立验证"
            with self._state_lock:
                self._training.update(
                    status="running",
                    message="训练已结束，正在原子保存模型和训练记录",
                )
            waiting_for_memory = stopped and memory_stop and not self._delete_after_stop
            run.update(
                status="stopped_deleted" if stopped and self._delete_after_stop else ("waiting_memory" if waiting_for_memory else ("stopped" if stopped else "completed")),
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                step=int(self._training.get("step", 0)), loss=self._training.get("loss"),
                message=("训练已暂停，内存恢复到安全水平后会自动继续" if waiting_for_memory else ("训练已停止，未完成模型已删除" if stopped and self._delete_after_stop else final_message)),
            )
            if stopped and self._delete_after_stop:
                self._remove_model_files(model_id)
                with self._state_lock:
                    self._training.update(status="stopped_deleted", message="训练已停止，未完成模型已删除")
            self._save_run(run)
            if not (stopped and self._delete_after_stop):
                self._write_model_metadata(model_id, metadata)
            with self._state_lock:
                self._training.update(
                    status="stopped_deleted" if stopped and self._delete_after_stop else ("stopped" if stopped else "completed"),
                    message=("训练已停止，未完成模型已删除" if stopped and self._delete_after_stop else final_message),
                )
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
        self._training_mode(payload)
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
        self._training_mode(payload)
        private_settings = self.teacher_settings()
        stored_rows = private_settings.get("profiles", {}).get(provider, [])
        requested_profile_id = str(payload.get("teacher_profile_id", "")).strip()
        active_profile_id = private_settings.get("active_profiles", {}).get(provider, "")
        stored_profile = next((row for row in stored_rows if row.get("id") == (requested_profile_id or active_profile_id)), {})
        api_key = str(payload.get("api_key", "")).strip() or str(stored_profile.get("api_key", "")).strip()
        selected_mode, _ = self._training_mode(payload, require_valid=False)
        policy = self._corpus_policy(selected_mode)
        manual_present = bool(str(payload.get("text", "")).strip())
        role_plan = (
            f"人工与 AI 分工（必须执行）：{policy['mixed_strategy']}"
            if manual_present else
            f"当前没有人工语料；AI 需要负责基础认知、通用语言能力和基础任务，并说明用户之后可以加入自己的专属文献。"
        )
        teacher_plan = "\n".join(filter(None, [str(payload.get("corpus_plan", "")).strip(), role_plan]))
        teacher = TeacherConfig(
            base_url=str(payload.get("teacher_base_url", "https://api.deepseek.com")),
            model=str(payload.get("teacher_model", "deepseek-v4-flash")),
            instruction=str(payload.get("instruction", "")), examples=int(payload.get("examples", 20)),
            language=str(payload.get("language", "中文")),
            corpus_mode=("fine_tuning" if self._training_mode(payload, require_valid=False)[0] == "fine_tuning" else "pretraining"),
            training_round=self._training_round(payload, str(payload.get("base_model", "")).strip() or None),
            corpus_plan=teacher_plan,
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
                "loss": None, "message": "正在调用教师 API 生成训练语料", "model_id": None,
                "teacher_model": teacher.model, "assisted": True,
                "generated_content_available": False, "generated_content_path": "",
                "generated_content_bytes": 0, "training_mode": selected_mode,
                "training_round": teacher.training_round,
                "_started_monotonic": time.monotonic(),
            }

        def generated(current: int, total: int) -> None:
            with self._state_lock:
                self._training.update(step=current, steps=total, message=f"正在生成训练语料：{current}/{total}")

        def worker() -> None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            partial_dataset = self.datasets_root / f"teacher-{stamp}.partial.txt"
            partial_tmp = partial_dataset.with_suffix(".tmp")
            partial_tmp.write_text(ORBIT_TRAINING_ANCHOR + "\n\n", encoding="utf-8")
            os.replace(partial_tmp, partial_dataset)

            def save_generated_chunk(chunk: str, current: int, total: int) -> None:
                # Append each completed provider response immediately. A
                # later 402/timeout/restart must leave the completed corpus
                # available for inspection and resume.
                with partial_dataset.open("a", encoding="utf-8") as handle:
                    handle.write(chunk.strip() + "\n\n")
                with self._state_lock:
                    self._training.update(
                        generated_content_available=True,
                        generated_content_path=str(partial_dataset),
                        generated_content_bytes=partial_dataset.stat().st_size,
                        dataset=str(partial_dataset),
                        message=f"正在生成训练语料：{current}/{total}（已保存）",
                    )
                self._save_generation_state()

            try:
                text, usage = generate_dataset(teacher, api_key, self._stop_event, generated, save_generated_chunk)
                if self._stop_event.is_set():
                    raise InterruptedError("自动训练已停止")
                dataset = self.datasets_root / f"teacher-{stamp}.txt"
                temporary = dataset.with_suffix(".tmp")
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, dataset)
                self.training_state_path.unlink(missing_ok=True)
                # Manual corpus and teacher corpus are additive. Keep the
                # teacher-only file for preview, and train on one combined
                # corpus when the user supplied both kinds of content.
                manual_text = str(payload.get("text", "")).strip()
                combined_text = (
                    f"{manual_text}\n\n--- ORBIT AI-ASSISTED CORPUS ---\n\n{text}"
                    if manual_text else text
                )
                combined_dataset = self.datasets_root / f"training-input-{stamp}.txt"
                combined_tmp = combined_dataset.with_suffix(".tmp")
                combined_tmp.write_text(combined_text, encoding="utf-8")
                os.replace(combined_tmp, combined_dataset)
                with self._state_lock:
                    self._training.update(
                        status="waiting_memory", message="样本已保存，正在等待足够的安全训练内存",
                        usage=usage, dataset=str(dataset), assisted=True,
                        generated_content_available=True, generated_content_path=str(dataset),
                        generated_content_bytes=len(text.encode("utf-8")),
                    )
                payload["_dataset_path"] = str(combined_dataset)
                payload["_generated_dataset_path"] = str(dataset)
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
                    delete_after_stop = bool(self._delete_after_stop)
                    pending = self._pending_training
                    dataset = self._training.get("dataset") or (pending[3] if pending else "")
                if delete_after_stop:
                    self._remove_training_dataset(dataset)
                with self._state_lock:
                    if delete_after_stop:
                        self._pending_training = None
                        self._delete_after_stop = False
                    self._training.update(
                        status="stopped_deleted" if delete_after_stop else "stopped",
                        message=("训练已停止，样本和未完成模型已删除" if delete_after_stop else str(exc)),
                        dataset="" if delete_after_stop else self._training.get("dataset", ""),
                        generated_content_available=False if delete_after_stop else self._training.get("generated_content_available", False),
                        generated_content_path="" if delete_after_stop else self._training.get("generated_content_path", ""),
                    )
                if delete_after_stop:
                    self.training_state_path.unlink(missing_ok=True)
                else:
                    self._save_generation_state()
            except Exception as exc:
                with self._state_lock:
                    delete_after_stop = bool(self._delete_after_stop)
                    pending = self._pending_training
                    dataset = self._training.get("dataset") or (pending[3] if pending else "")
                if delete_after_stop:
                    self._remove_training_dataset(dataset)
                with self._state_lock:
                    if delete_after_stop:
                        self._pending_training = None
                        self._delete_after_stop = False
                    self._training.update(
                        status="stopped_deleted" if delete_after_stop else "failed",
                        message=("训练已停止，样本和未完成模型已删除" if delete_after_stop else (
                            f"{exc}；已完成 {self._training.get('step', 0)}/{self._training.get('steps', 0)} 个样本，已生成内容保存在本机，可在训练页查看"
                            if self._training.get("generated_content_available") else str(exc)
                        )),
                        dataset="" if delete_after_stop else self._training.get("dataset", ""),
                        generated_content_available=False if delete_after_stop else self._training.get("generated_content_available", False),
                        generated_content_path="" if delete_after_stop else self._training.get("generated_content_path", ""),
                    )
                if delete_after_stop:
                    self.training_state_path.unlink(missing_ok=True)
                elif self._training.get("generated_content_available"):
                    self._save_generation_state()

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
            "training_mode": str(run.get("training_mode") or ("fine_tuning" if run.get("parent_model") else "pretraining")),
            "training_round": int(run.get("training_round", 2 if run.get("parent_model") else 1) or 1),
            "base_model": str(run.get("parent_model") or ""), "device": str(run.get("device", "auto")),
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
            training_mode=str(payload.get("training_mode", "pretraining")),
            training_round=int(payload.get("training_round", 1) or 1),
            base_model=str(payload.get("base_model", "")),
            optimization_goal=str(payload.get("optimization_goal", "balanced")),
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
        if model_id and (Path(model_id).name != model_id or model_id in {"", ".", ".."}):
            raise ValueError("无效的模型名称")

        with self._state_lock:
            if self._hub_upload.get("model") == model_id and self._hub_upload.get("status") in {"hashing", "uploading"}:
                raise RuntimeError("模型正在上传，暂时不能删除")

        # A user may have loaded the checkpoint after stopping.  Reuse the
        # normal unload path before removing the exact model files.
        if model_id and self.active_model_id == model_id:
            self.unload_model()
        removed = self._remove_model_files(model_id) if model_id else []

        # AI generation and memory-wait states can end before a checkpoint or
        # training run exists.  They still need a reliable delete action: only
        # remove temporary datasets inside Orbit's dataset directory.
        with self._state_lock:
            pending = self._pending_training
            state_dataset = state.get("dataset")
        candidates = [state_dataset]
        if pending:
            candidates.append(pending[3])
        for candidate in candidates:
            if self._remove_training_dataset(candidate):
                removed.append(Path(str(candidate)).name)

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
            self._delete_after_stop = False
            self._training.update(
                status="stopped_deleted",
                message="训练已停止，样本和未完成模型已删除",
                checkpoint="", dataset="", generated_content_available=False,
                generated_content_path="", generated_content_bytes=0,
            )
        return {
            "status": "stopped_deleted",
            "model": model_id,
            "removed": removed,
            "training_history_preserved": True,
        }

    def _remove_training_dataset(self, value: Any) -> bool:
        """Remove only a temporary training dataset owned by Orbit."""
        if not value:
            return False
        candidate = Path(str(value))
        try:
            if candidate.is_file() and candidate.resolve().parent == self.datasets_root.resolve():
                candidate.unlink()
                return True
        except OSError:
            pass
        return False

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
            # unload_model() sets the cancellation event for an existing
            # loader.  This is a new load operation, so clear that stale flag
            # after unloading and before starting the inference worker.
            self._load_cancel.clear()
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
                "temperature": temperature,
                "memory_context": memory_context,
                "model_name": metadata.get("name", self._model_id),
            }, ensure_ascii=False) + "\n")
            self._model.stdin.flush()
            readable, _, _ = select.select([self._model.stdout], [], [], 45.0)
            if not readable:
                process = self._model
                self._model = None
                self._model_id = None
                self._model_device = "cpu"
                try:
                    process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self._set_loading("idle", 0, "模型推理超时，已自动释放推理进程", None)
                raise RuntimeError("本次回复超过 45 秒未返回，已自动释放模型；请重试或减少问题长度")
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
