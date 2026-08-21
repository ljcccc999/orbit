from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import psutil


GB = 1_000_000_000
TRAINING_MEMORY_RESERVE_GB = 1.0


def resource_snapshot(path: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(path)
    return {
        "memory_total_gb": round(memory.total / GB, 1),
        "memory_available_gb": round(memory.available / GB, 1),
        "memory_percent": round(float(memory.percent), 1),
        "disk_free_gb": round(disk.free / GB, 1),
        "disk_total_gb": round(disk.total / GB, 1),
    }


def require_training_capacity(required_memory_gb: float, parameter_count: int, path: Path) -> dict[str, Any]:
    snapshot = resource_snapshot(path)
    # This is the free-memory floor for the OS and Orbit. Model training
    # memory is checked separately above this reserve.
    memory_reserve = TRAINING_MEMORY_RESERVE_GB
    usable_memory = max(0.0, snapshot["memory_available_gb"] - memory_reserve)
    if required_memory_gb > usable_memory:
        raise MemoryError(
            f"训练预计需要约 {required_memory_gb:.1f}GB，目前可用内存约 "
            f"{snapshot['memory_available_gb']:.1f}GB；Orbit 已阻止本次分配，避免系统因内存耗尽而失去响应。"
        )
    checkpoint_gb = max(2.0, parameter_count * 16 / GB)
    if snapshot["disk_free_gb"] < checkpoint_gb + 2.0:
        raise OSError(
            f"磁盘空间不足：至少需要约 {checkpoint_gb + 2.0:.1f}GB 可用空间，"
            f"当前约 {snapshot['disk_free_gb']:.1f}GB。"
        )
    return snapshot


def require_inference_capacity(parameter_count: int, path: Path) -> dict[str, Any]:
    snapshot = resource_snapshot(path)
    required = max(1.0, parameter_count * 5.5 / GB)
    if required + 1.5 > snapshot["memory_available_gb"]:
        raise MemoryError(
            f"加载模型预计需要约 {required:.1f}GB，目前可用内存约 "
            f"{snapshot['memory_available_gb']:.1f}GB。为防止电脑失去响应，Orbit 没有加载该模型。"
        )
    return snapshot


def require_checkpoint_load_capacity(checkpoint_size: int, path: Path) -> dict[str, Any]:
    snapshot = resource_snapshot(path)
    # Existing checkpoints may contain optimizer state as well as weights.
    # Loading in a worker temporarily needs both serialized state and a model.
    required = max(1.0, checkpoint_size * 1.5 / GB)
    if required + 1.5 > snapshot["memory_available_gb"]:
        raise MemoryError(
            f"加载该 checkpoint 预计需要约 {required:.1f}GB 临时内存，目前可用约 "
            f"{snapshot['memory_available_gb']:.1f}GB。Orbit 已阻止加载以保护系统。"
        )
    return snapshot


def memory_is_critical() -> bool:
    memory = psutil.virtual_memory()
    reserve = int(TRAINING_MEMORY_RESERVE_GB * GB)
    return memory.available < reserve
