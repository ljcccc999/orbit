"""Checkpoint helpers for MLX models."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .config import OrbitConfig


def parameter_arrays(model):
    """Return stable flat names and arrays suitable for safetensors."""
    return {name: value for name, value in tree_flatten(model.parameters())}


def parameter_count(model) -> int:
    return sum(array.size for array in parameter_arrays(model).values())


def save_checkpoint(model, config: OrbitConfig, directory: str | Path, *, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = parameter_arrays(model)
    mx.save_safetensors(str(directory / "weights.safetensors"), arrays)
    info = {
        "model": "Orbit",
        "architecture": "Orbit Continuum Architecture",
        "format": "mlx-safetensors",
        "config": config.__dict__,
        "parameters": sum(array.size for array in arrays.values()),
        "dtype": str(next(iter(arrays.values())).dtype),
        "note": "For quantized checkpoints, `parameters` is packed storage elements; use config/estimated_parameters for logical weights.",
    }
    if metadata:
        info.update(metadata)
    (directory / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
