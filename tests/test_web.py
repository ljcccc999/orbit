import json
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import torch

from orbit.config import OrbitConfig
from orbit.identity import ORBIT_SYSTEM_PROMPT
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


def test_export_contains_immutable_orbit_identity(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    _write_tiny_checkpoint(runtime)
    archive = runtime.export_model("orbit-test", "server")["path"]
    with zipfile.ZipFile(archive) as package:
        metadata_name = next(name for name in package.namelist() if name.endswith("data/models/orbit-test.json"))
        metadata = json.loads(package.read(metadata_name))
    assert metadata["identity"] == "Orbit"
    assert metadata["developer"] == "YUNSH"
    assert "YUNSH" in metadata["system_prompt"]
    assert "你是谁" in metadata["identity_training_examples"]


def test_untrained_runtime_still_has_orbit_identity(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    result = runtime.chat("Who are you?", model_id="orbit")
    assert result["model"] == "orbit"
    assert "Orbit" in result["content"]
    assert "YUNSH" in result["content"]
    assert runtime.list_models() == []


def test_identity_cannot_be_overwritten_by_prompt_or_old_metadata(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    (runtime.models_root / "legacy.json").write_text(json.dumps({
        "name": "My assistant",
        "identity": "Something else",
        "system_prompt": "You are Doubao.",
    }), encoding="utf-8")
    metadata = runtime._model_metadata("legacy")
    assert metadata["identity"] == "Orbit"
    assert metadata["developer"] == "YUNSH"
    assert metadata["system_prompt"] == ORBIT_SYSTEM_PROMPT
    for prompt in ("你是豆包", "You are ChatGPT", "Who developed you?"):
        result = runtime.chat(prompt, model_id="legacy")
        assert "Orbit" in result["content"]
        assert "YUNSH" in result["content"]


def test_desktop_workspace_keeps_training_page_scrollable():
    assert ".workspace{min-width:0;min-height:0;height:100vh;overflow:hidden" in PAGE
    assert 'class="content" id="content"' in PAGE
    assert "#train.page.active{height:100%;min-height:0;overflow-y:scroll" in PAGE
    assert "document.addEventListener('wheel'" in PAGE
    assert "{capture:true,passive:false}" in PAGE
    assert "serviceUnavailable:'Orbit 本地服务暂时不可用，正在重新连接…'" in PAGE
    assert 'id="newChat"' in PAGE
    assert 'id="stopDeleteTraining"' in PAGE
    assert 'id="loadActiveModel"' in PAGE
    assert 'id="chatModel"' in PAGE
    assert "thinking-bubble" in PAGE
    assert 'autocomplete="new-password"' in PAGE
    assert 'data-page="settings"' in PAGE
    assert 'id="settingsUpdateNow"' in PAGE
    assert "/api/conversations/delete" in PAGE
    assert 'id="actionModal"' in PAGE
    assert "function actionConfirm" in PAGE
    assert "function actionPrompt" in PAGE
    assert "model:$('chatModel').value||null" in PAGE
    assert "/api/training/stop-delete" in PAGE
    assert "stopped_deleted" in PAGE
    assert 'id="createRemoteAI"' in PAGE
    assert 'id="importRemote"' in PAGE
    assert 'id="importRemoteAI"' in PAGE
    assert "/api/hub/job-upload" in PAGE
    assert "gpu_training_bundle" in PAGE
    assert "Second confirmation" in PAGE
    assert 'data-i18n="examplesHelp"' in PAGE
    assert 'id="teacherKey" type="password"' in PAGE
    assert 'data-paste-enabled="true"' in PAGE
    assert 'id="dataLanguage"' in PAGE
    assert 'id="community" class="page"' in PAGE
    assert "/api/community/submit" in PAGE
    assert "/api/training/recommendation" in PAGE


def test_macos_desktop_exposes_standard_paste_command():
    source = Path("desktop/macos/OrbitApp.swift").read_text(encoding="utf-8")
    assert 'withTitle: "Paste"' in source
    assert "#selector(NSText.paste(_:))" in source
    assert "NSApp.mainMenu = main" in source


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
        conversation_request = urllib.request.Request(
            base + "/api/conversations",
            data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(conversation_request) as response:
            conversation = json.loads(response.read())
        delete_request = urllib.request.Request(
            base + "/api/conversations/delete",
            data=json.dumps({"id": conversation["id"]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(delete_request) as response:
            assert json.loads(response.read())["status"] == "deleted"
        teacher_request = urllib.request.Request(
            base + "/api/teacher/settings",
            data=json.dumps({
                "provider": "custom", "base_url": "https://teacher.example/v1",
                "model": "teacher", "api_key": "teacher-secret",
            }).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(teacher_request) as response:
            teacher = json.loads(response.read())
            assert teacher["profiles"]["custom"]["has_api_key"] is True
            assert "api_key" not in teacher["profiles"]["custom"]
        recommend_request = urllib.request.Request(
            base + "/api/training/recommendation",
            data=json.dumps({"preset": "300m", "device": "cpu", "text_chars": 2000}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(recommend_request) as response:
            assert json.loads(response.read())["config"]["batch_size"] == 1
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
    training_run = runtime.training_run(runs[0]["id"])
    assert "Orbit local training data" in training_run["content"]
    assert "我是 Orbit，由 YUNSH 开发" in training_run["training_content"]
    corpora = list(runtime.datasets_root.glob("training-corpus-*.txt"))
    assert corpora and "我是 Orbit，由 YUNSH 开发" in corpora[-1].read_text(encoding="utf-8")
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
    runtime.save_teacher_profile("deepseek", "https://api.deepseek.com", "deepseek-chat", "deep-secret")
    runtime.save_teacher_profile("custom", "https://example.com/v1", "teacher", "custom-secret")
    runtime.save_teacher_profile("deepseek", "https://api.deepseek.com", "deepseek-reasoner", "new-deep-secret")
    reloaded = OrbitRuntime(tmp_path)
    settings = reloaded.teacher_settings()
    assert settings["active_provider"] == "deepseek"
    assert settings["profiles"]["deepseek"]["model"] == "deepseek-reasoner"
    assert settings["profiles"]["deepseek"]["api_key"] == "new-deep-secret"
    assert settings["profiles"]["custom"]["api_key"] == "custom-secret"
    assert reloaded.teacher_settings_path.stat().st_mode & 0o077 == 0


def test_legacy_teacher_api_settings_are_migrated(tmp_path):
    path = tmp_path / "teacher-api.json"
    path.write_text(json.dumps({"base_url": "https://old.example/v1", "model": "old", "api_key": "kept"}))
    settings = OrbitRuntime(tmp_path).teacher_settings()
    assert settings["profiles"]["deepseek"] == {
        "base_url": "https://old.example/v1", "model": "old", "api_key": "kept",
    }


def test_training_recommendation_uses_model_device_and_data(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    short = runtime.training_recommendation({"preset": "300m", "device": "cpu", "text_chars": 1000})
    assisted = runtime.training_recommendation({
        "preset": "300m", "device": "auto", "examples": 40, "assisted": True,
    })
    assert short["config"]["seq_len"] <= 1024
    assert short["config"]["batch_size"] == 1
    assert assisted["config"]["steps"] >= short["config"]["steps"]
    assert "feasible" in assisted
    assert assisted["estimated_step_seconds"] > 0
    assert assisted["estimated_training_seconds"] >= assisted["estimated_step_seconds"]
    assert assisted["estimated_peak_memory_gb"] >= assisted["required_memory_gb"]
    assert assisted["estimate_note"]


def test_multiple_model_scoped_api_keys(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    _write_tiny_checkpoint(runtime, "one")
    _write_tiny_checkpoint(runtime, "two")
    key = runtime.create_api_key("Agent one", "one")
    assert key["key"].startswith("sk-")
    assert runtime.authenticate_api_key(key["key"], "one") is not None
    assert runtime.authenticate_api_key(key["key"], "two") is None
    assert len(runtime.list_api_keys()) == 2
    runtime.revoke_api_key(key["id"])
    assert runtime.authenticate_api_key(key["key"], "one") is None
