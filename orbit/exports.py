from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from .identity import ORBIT_SYSTEM_PROMPT


def _zip_directory(root: Path, output: Path) -> None:
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in root.rglob("*"):
            if file.is_file():
                archive.write(file, arcname=f"{root.name}/{file.relative_to(root)}")
    os.replace(temporary, output)


def _write_native_server(project_root: Path, build: Path, model_id: str, checkpoint: Path, metadata: dict[str, Any]) -> None:
    shutil.copy2(project_root / "pyproject.toml", build / "pyproject.toml")
    shutil.copytree(project_root / "orbit", build / "orbit")
    models = build / "data" / "models"
    models.mkdir(parents=True)
    shutil.copy2(checkpoint, models / checkpoint.name)
    (models / f"{model_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (build / "run.sh").write_text(
        "#!/bin/sh\nset -eu\nROOT=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        "python3 -m venv \"$ROOT/.venv\"\n\"$ROOT/.venv/bin/python\" -m pip install \"$ROOT\"\n"
        "exec \"$ROOT/.venv/bin/python\" -m orbit.web --no-browser --host \"${ORBIT_HOST:-127.0.0.1}\" "
        "--port \"${ORBIT_PORT:-8765}\" --data-dir \"$ROOT/data\"\n",
        encoding="utf-8",
    )
    os.chmod(build / "run.sh", 0o755)
    (build / "run.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n$Root = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
        "py -3 -m venv \"$Root\\.venv\"\n& \"$Root\\.venv\\Scripts\\python.exe\" -m pip install $Root\n"
        "& \"$Root\\.venv\\Scripts\\python.exe\" -m orbit.web --no-browser --host 127.0.0.1 --port 8765 --data-dir \"$Root\\data\"\n",
        encoding="utf-8",
    )
    (build / "README.md").write_text(
        f"# {model_id} · Orbit server export\n\n"
        "This package contains the selected checkpoint, its Orbit identity and the OpenAI-compatible local server. "
        "The model runs locally and does not call a cloud model during inference.\n\n"
        "- macOS/Linux: `./run.sh`\n"
        "- Windows PowerShell: `./run.ps1`\n"
        "- API base URL: `http://127.0.0.1:8765/v1`\n"
        "- Open `http://127.0.0.1:8765` to copy the randomly generated API key.\n\n"
        "Python dependencies may need to be installed once before the server can run fully offline. "
        "Set `ORBIT_HOST=0.0.0.0` only on a server protected by an appropriate firewall.\n",
        encoding="utf-8",
    )


def _write_ollama(build: Path, model_id: str, gguf: Path, metadata: dict[str, Any]) -> None:
    shutil.copy2(gguf, build / "model.gguf")
    system_prompt = ORBIT_SYSTEM_PROMPT
    escaped_prompt = system_prompt.replace('"""', '\\\"\\\"\\\"')
    ollama_name = "".join(ch.lower() if ch.isascii() and (ch.isalnum() or ch in "._-") else "-" for ch in model_id)
    ollama_name = "-".join(part for part in ollama_name.split("-") if part) or "orbit-model"
    (build / "Modelfile").write_text(
        f"FROM ./model.gguf\nSYSTEM \"\"\"{escaped_prompt} The current model name is {model_id}.\"\"\"\n"
        "PARAMETER temperature 0.8\n",
        encoding="utf-8",
    )
    (build / "README.md").write_text(
        f"# {model_id} · Ollama export\n\n"
        f"Run `ollama create {ollama_name} -f Modelfile`, then `ollama run {ollama_name}`. "
        f"The Orbit display name remains `{model_id}`.\n\n"
        "This export is available only when a compatible GGUF file has been attached to the model. "
        "Orbit's native `.pt` hybrid architecture cannot be presented to Ollama as if it were GGUF.\n",
        encoding="utf-8",
    )


def create_model_export(
    project_root: Path, data_root: Path, model_id: str, checkpoint: Path,
    metadata: dict[str, Any], target: str,
) -> Path:
    if target not in {"server", "ollama"}:
        raise ValueError("导出目标必须是 server 或 ollama")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    exports_root = data_root / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    build = exports_root / f"{model_id}-{target}-{stamp}"
    build.mkdir(parents=False, exist_ok=False)
    try:
        if target == "server":
            _write_native_server(project_root, build, model_id, checkpoint, metadata)
        else:
            gguf = checkpoint.with_suffix(".gguf")
            if not gguf.is_file():
                raise ValueError(
                    "这个模型是 Orbit 原生 .pt 架构，Ollama 不能直接读取。请先使用服务器导出；"
                    "只有为该模型提供兼容的同名 .gguf 文件后，才能生成 Ollama 包。"
                )
            _write_ollama(build, model_id, gguf, metadata)
        output = exports_root / f"{build.name}.zip"
        _zip_directory(build, output)
        return output
    finally:
        shutil.rmtree(build, ignore_errors=True)
