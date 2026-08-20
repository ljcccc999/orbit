import json
import threading
import time
import urllib.request

import torch

from orbit.config import OrbitConfig
from orbit.model import OrbitForCausalLM
from orbit.runtime import OrbitRuntime
from orbit.web import OrbitHTTPServer


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
        "text": "Orbit local training data. " * 10,
    })
    assert state["status"] in {"preparing", "running"}
    deadline = time.time() + 20
    while runtime.training_state()["status"] in {"preparing", "running", "stopping"} and time.time() < deadline:
        time.sleep(0.05)
    finished = runtime.training_state()
    assert finished["status"] == "completed"
    assert runtime.list_models()[0]["id"] == finished["model_id"]
    assert runtime._training_process is None


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
