import json
import threading
import time
import urllib.request
import zipfile

import torch

from orbit.config import OrbitConfig
from orbit.model import OrbitForCausalLM
from orbit.runtime import OrbitRuntime
from orbit.web import OrbitHTTPServer
from orbit.web_ui import PAGE


def _write_tiny_checkpoint(runtime: OrbitRuntime, model_id: str = "orbit-test") -> None:
    cfg = OrbitConfig.tiny().with_overrides(n_layers=1, max_seq_len=32)
    model = OrbitForCausalLM(cfg)
    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, runtime.models_root / f"{model_id}.pt")


def test_runtime_loads_and_serves_local_checkpoint(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    _write_tiny_checkpoint(runtime)
    assert runtime.list_models()[0]["id"] == "orbit-test"
    result = runtime.chat("hello", model_id="orbit-test", max_tokens=2, temperature=0)
    assert result["model"] == "orbit-test"
    assert isinstance(result["content"], str)


def test_untrained_runtime_still_has_orbit_identity(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    result = runtime.chat("Who are you?", model_id="orbit")
    assert result["model"] == "orbit"
    assert "Orbit" in result["content"]
    assert runtime.list_models() == []


def test_desktop_workspace_keeps_training_page_scrollable():
    assert ".workspace{min-width:0;min-height:0" in PAGE
    assert ".content{min-height:0;overflow-x:hidden;overflow-y:auto" in PAGE
    assert "serviceUnavailable:'Orbit 本地服务暂时不可用，正在重新连接…'" in PAGE


def test_http_health_models_and_openai_chat(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    _write_tiny_checkpoint(runtime)
    server = OrbitHTTPServer(("127.0.0.1", 0), runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/health") as response:
            assert json.loads(response.read())["name"] == "orbit"
        headers = {"Authorization": f"Bearer {runtime.local_api_key}"}
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=headers)) as response:
            assert json.loads(response.read())["data"][0]["id"] == "orbit-test"
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "orbit-test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "temperature": 0,
            }).encode(),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
            assert payload["object"] == "chat.completion"
            assert payload["model"] == "orbit-test"
        request = urllib.request.Request(
            base + "/v1/responses",
            data=json.dumps({
                "model": "orbit-test",
                "input": "hello",
                "max_output_tokens": 1,
                "temperature": 0,
            }).encode(),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
            assert payload["object"] == "response"
            assert payload["status"] == "completed"
            assert payload["model"] == "orbit-test"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_training_creates_a_local_checkpoint(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    state = runtime.start_training({
        "preset": "local",
        "steps": 1,
        "batch_size": 1,
        "seq_len": 8,
        "grad_accum": 1,
        "learning_rate": 3e-4,
        "warmup_steps": 0,
        "checkpoint_every": 0,
        "device": "cpu",
        "model_name": "my-orbit",
        "text": "Orbit local training data. " * 10,
    })
    assert state["status"] in {"preparing", "running"}
    deadline = time.time() + 20
    while runtime.training_state()["status"] in {"preparing", "running", "stopping"} and time.time() < deadline:
        time.sleep(0.05)
    finished = runtime.training_state()
    assert finished["status"] == "completed"
    assert runtime.list_models()[0]["id"] == "my-orbit"
    runs = runtime.list_training_runs()
    assert len(runs) == 1
    assert runs[0]["model_id"] == "my-orbit"
    assert "Orbit local training data" in runtime.training_run(runs[0]["id"])["content"]
    assert runtime._training_process is None


def test_secondary_training_records_parent_and_server_export(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    payload = {
        "preset": "local", "steps": 1, "batch_size": 1, "seq_len": 8,
        "grad_accum": 1, "learning_rate": 3e-4, "warmup_steps": 0,
        "checkpoint_every": 0, "device": "cpu", "model_name": "base",
        "text": "Orbit secondary training data. " * 10,
    }
    runtime.start_training(payload)
    deadline = time.time() + 20
    while runtime.training_state()["status"] in {"preparing", "running", "stopping"} and time.time() < deadline:
        time.sleep(0.05)
    assert runtime.training_state()["status"] == "completed"
    runtime.start_training({**payload, "model_name": "child", "base_model": "base"})
    deadline = time.time() + 20
    while runtime.training_state()["status"] in {"preparing", "running", "stopping"} and time.time() < deadline:
        time.sleep(0.05)
    assert runtime.training_state()["status"] == "completed"
    child = next(row for row in runtime.list_models() if row["id"] == "child")
    assert child["parent_model"] == "base"
    assert len(child["training_runs"]) == 2

    exported = runtime.export_model("child", "server")
    with zipfile.ZipFile(exported["path"]) as archive:
        names = archive.namelist()
        assert any(name.endswith("/data/models/child.pt") for name in names)
        assert any(name.endswith("/data/models/child.json") for name in names)
        assert any(name.endswith("/run.sh") for name in names)
    try:
        runtime.export_model("child", "ollama")
    except ValueError as exc:
        assert "GGUF" in str(exc) or ".gguf" in str(exc)
    else:
        raise AssertionError("native checkpoint must not be mislabeled as Ollama-compatible")


def test_teacher_api_settings_persist_locally(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    runtime._save_teacher_settings({"base_url": "https://example.com/v1", "model": "teacher", "api_key": "secret"})
    reloaded = OrbitRuntime(tmp_path)
    assert reloaded.teacher_settings()["api_key"] == "secret"
    assert reloaded.teacher_settings_path.stat().st_mode & 0o077 == 0


def test_multiple_model_scoped_api_keys(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    _write_tiny_checkpoint(runtime, "one")
    _write_tiny_checkpoint(runtime, "two")
    key = runtime.create_api_key("Agent one", "one")
    assert runtime.authenticate_api_key(key["key"], "one") is not None
    assert runtime.authenticate_api_key(key["key"], "two") is None
    assert len(runtime.list_api_keys()) == 2
    runtime.revoke_api_key(key["id"])
    assert runtime.authenticate_api_key(key["key"], "one") is None
