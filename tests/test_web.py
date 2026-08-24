import json
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import orbit.runtime as runtime_module

from orbit.config import OrbitConfig
from orbit.identity import ORBIT_SYSTEM_PROMPT
from orbit.model import OrbitForCausalLM
from orbit.runtime import OrbitRuntime
from orbit.web import OrbitHTTPServer
from orbit.web_ui import PAGE


def test_training_page_preserves_a_manually_entered_sample_count():
    assert 'max="10000000"' in PAGE
    assert "manualSampleEdit=true" in PAGE
    assert "if(!manualSampleEdit)$('teacherExamples').value=r.recommended_examples" in PAGE
    assert "use_recommended_examples:!manualSampleEdit" in PAGE
    assert "步 · 序列" in PAGE
    assert "个样本" in PAGE
    assert "viewGeneratedContent" in PAGE
    assert "tokenFinalResult" in PAGE
    assert "tokenResearchStandard" in PAGE
    assert "越接近 0%" in PAGE
    assert "最终计算结果" in PAGE


def test_reset_form_balanced_recommendation_is_coupled_to_recommended_samples(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    recommendation = runtime.training_recommendation({
        "preset": "300m",
        "device": "auto",
        "examples": 20,
        "training_mode": "pretraining",
        "training_round": 1,
        "optimization_goal": "balanced",
        "use_recommended_examples": True,
    })
    assert recommendation["optimization_goal"] == "balanced"
    # The assistant field follows the Chinchilla-sized reference here; larger
    # models may exceed the per-round safety cap and must use multiple rounds.
    assert recommendation["recommended_examples"] == 888_889
    assert recommendation["required_examples_for_reference"] == 888_889
    assert recommendation["config"]["steps"] > 100_000
    estimate = recommendation["training_advice"]["dataset_estimate"]
    assert estimate["target_pretraining_tokens"] > 5_000_000_000
    assert 99.9 <= estimate["target_coverage_percent"] <= 100.1
    assert recommendation["token_planning"]["tokens_per_parameter_reference"] == 20
    assert recommendation["training_advice"]["chat_ready_recipe"]["curated_reference_samples"] == 1_000

    fast = runtime.training_recommendation({
        "preset": "300m",
        "device": "auto",
        "examples": 20,
        "training_mode": "pretraining",
        "training_round": 1,
        "optimization_goal": "fast",
        "use_recommended_examples": True,
    })
    memory = runtime.training_recommendation({
        "preset": "300m",
        "device": "auto",
        "examples": 20,
        "training_mode": "pretraining",
        "training_round": 1,
        "optimization_goal": "memory",
        "use_recommended_examples": True,
    })
    assert fast["estimated_training_seconds"] <= recommendation["estimated_training_seconds"]
    assert memory["config"]["seq_len"] <= recommendation["config"]["seq_len"]
    assert "applyOptimizationRecommendation" in PAGE
    assert "$('optimizationGoal')?.addEventListener('change',()=>{manualConfigEdit=false;scheduleRecommendation()})" in PAGE


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
    try:
        runtime.chat("Who are you?", model_id="orbit")
    except FileNotFoundError as exc:
        assert "找不到本地模型" in str(exc)
    else:
        raise AssertionError("an untrained runtime must not use a hard-coded identity answer")
    assert runtime.list_models() == []


def test_stopped_training_can_delete_model_and_keep_history(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    model_id = "stopped-model"
    run_id = "run-stopped-model"
    checkpoint = runtime.models_root / f"{model_id}.pt"
    checkpoint.write_bytes(b"checkpoint")
    (runtime.models_root / f"{model_id}.json").write_text("{}", encoding="utf-8")
    runtime._save_run({
        "id": run_id, "status": "stopped", "model_id": model_id,
        "model_name": model_id, "dataset": "", "training_dataset": "",
    })
    with runtime._state_lock:
        runtime._training.update(
            status="stopped", model_id=model_id, run_id=run_id,
            checkpoint=str(checkpoint), message="用户已请求安全停止",
        )

    result = runtime.stop_training(delete_checkpoint=True)

    assert result["status"] == "stopped_deleted"
    assert not checkpoint.exists()
    assert not (runtime.models_root / f"{model_id}.json").exists()
    assert runtime.training_run(run_id)["status"] == "stopped_deleted"


def test_teacher_generated_content_is_available_only_from_orbit_dataset_dir(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    dataset = runtime.datasets_root / "teacher-preview.txt"
    dataset.write_text("文献样本\n\n输入：查询年龄大于18岁的用户\n输出：SELECT * FROM users WHERE age > 18", encoding="utf-8")
    with runtime._state_lock:
        runtime._training.update(
            status="waiting_memory", assisted=True,
            generated_content_available=True,
            generated_content_path=str(dataset),
            generated_content_bytes=dataset.stat().st_size,
        )

    result = runtime.generated_training_content()

    assert result["available"] is True
    assert "SELECT * FROM users" in result["content"]
    assert result["assisted"] is True

    outside = tmp_path / "outside.txt"
    outside.write_text("must not be exposed", encoding="utf-8")
    with runtime._state_lock:
        runtime._training["generated_content_path"] = str(outside)
    assert runtime.generated_training_content()["available"] is False


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
    try:
        runtime.chat("Who developed you?", model_id="legacy")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("metadata alone must not become a hard-coded runtime answer")


def test_desktop_workspace_keeps_training_page_scrollable():
    assert ".workspace{min-width:0;min-height:0;height:100vh;overflow:hidden" in PAGE
    assert 'class="content" id="content"' in PAGE
    assert "#train.page.active{height:100%;min-height:0;overflow-y:scroll" in PAGE
    assert "document.addEventListener('wheel'" in PAGE
    assert "{capture:true,passive:false}" in PAGE
    assert "serviceUnavailable:'Orbit 本地服务暂时不可用，正在重新连接…'" in PAGE
    assert 'id="newChat"' in PAGE
    assert 'id="stopDeleteTraining"' in PAGE
    assert 'id="continueTrainingButton"' in PAGE
    assert 'id="loadActiveModel"' in PAGE
    assert 'id="chatModel"' in PAGE
    assert "thinking-bubble" in PAGE
    assert 'autocomplete="new-password"' not in PAGE
    assert 'id="teacherKey" type="password" autocomplete="off" data-paste-enabled="true"' in PAGE
    assert 'data-page="settings"' in PAGE
    assert 'id="settingsUpdateNow"' in PAGE
    assert "/api/conversations/archive" in PAGE
    assert 'id="actionModal"' in PAGE
    assert "function actionConfirm" in PAGE
    assert 'id="generatedContentPanel"' in PAGE
    assert 'id="generatedContentPreview" class="code training-content" tabindex="0"' in PAGE
    assert 'id="viewGeneratedContent"' in PAGE
    assert 'id="copyGeneratedContent"' in PAGE
    assert "/api/training/generated-content" in PAGE
    assert "AI-generated content" in PAGE
    assert "function actionPrompt" in PAGE
    assert 'id="installUpdates"' in PAGE
    assert "function installFromSidebar" in PAGE
    assert "/api/update/install" in PAGE
    assert "PAGE = PAGE.replace" not in PAGE
    assert 'id="memoryInput"' in PAGE
    assert 'id="saveMemory"' in PAGE
    assert "Orbit ${system.resources.orbit_memory_gb" in PAGE
    assert "function renderTeacherProfiles" in PAGE
    assert "model:$('chatModel').value||null" in PAGE
    assert "/api/training/stop-delete" in PAGE
    assert "data-delete-run" in PAGE
    assert "data-continue-run" in PAGE
    assert "Training history deleted." in PAGE
    assert "function resetTrainingForm" in PAGE
    assert "actionConfirm(t('deleteProfile')" in PAGE
    assert "stopped_deleted" in PAGE
    assert 'id="createRemoteAI"' in PAGE
    assert 'id="importRemote"' not in PAGE
    assert 'id="importRemoteAI"' not in PAGE
    assert "Training history deleted." in PAGE
    assert 'data-i18n="examplesHelp"' in PAGE
    assert 'id="teacherKey" type="password"' in PAGE
    assert 'data-paste-enabled="true"' in PAGE
    assert 'id="dataLanguage"' in PAGE
    assert 'id="trainingMode"' in PAGE
    assert 'data-i18n="fineTuning"' in PAGE
    assert "function syncTrainingMode" in PAGE
    assert "system.models[0].id" in PAGE
    assert 'id="baseModelSource"' in PAGE
    assert 'id="downloadBasePanel"' in PAGE
    assert "function syncBaseModelChoice" in PAGE
    assert "External models cannot be trained from scratch" in PAGE
    assert "base_model_source:$('baseModelSource')?.value||'local'" in PAGE
    assert "/api/models/download" in PAGE
    assert 'id="community" class="page"' in PAGE
    assert "/api/community/submit" in PAGE
    assert "/api/training/recommendation" in PAGE


def test_orbit_code_uses_one_scrollable_conversation_shell():
    assert 'id="toggleSidebar"' in PAGE
    assert 'id="navigateBack"' in PAGE
    assert 'id="navigateForward"' in PAGE
    assert 'sidebar-top-controls' not in PAGE.split("<body>", 1)[1]
    assert '#content.code-page-active{overflow:hidden!important' in PAGE
    assert '#code .code-timeline{min-height:0;overflow-x:hidden;overflow-y:auto' in PAGE
    assert '.code-process-bar,.code-process-overview,.current-activity{display:none!important}' in PAGE
    assert '#codeProgress:not([hidden]){display:flex!important}' in PAGE
    assert "function codeInlineProgress(){return''}" in PAGE
    assert 'class="code-process-inline" hidden' in PAGE
    assert 'id="composerAdvancedToggle"' in PAGE
    assert 'const codeSourceField=' in PAGE
    assert "$('codeAdvanced').textContent=orbitShortModelName()" in PAGE
    assert "function configureContextualSidebar(mode)" in PAGE
    assert ".sidebar.mode-training>.nav button[data-page=\"community\"]" in PAGE
    assert ".sidebar.mode-orbit>.nav button[data-page=\"community\"]" not in PAGE
    assert "orbitWorkspaceMode!=='code'" in PAGE
    assert "showPage('codeApi')" in PAGE
    assert "调用了 ${tools.length} 个工具" in PAGE
    assert "正在调用':'调用了" in PAGE
    assert '#chat .chat-panel>.panel-head,#chat .inspector{display:none!important}' in PAGE
    assert '#chat .composer{position:absolute' in PAGE
    assert "originalWorkspace=$('codeWorkspace')?.closest('.field')" in PAGE
    assert "list.appendChild(originalWorkspace)" in PAGE
    assert "showPage('code');newCodeSession()" in PAGE
    assert "--orbit-spring:cubic-bezier(.2,.82,.22,1)" in PAGE
    assert "@supports(corner-shape:squircle)" in PAGE
    assert "backdrop-filter:blur(48px) saturate(165%)" in PAGE
    assert "#codeApi.page.active" in PAGE
    assert ".product-menu:not([hidden]){animation:orbitGlassIn" in PAGE
    assert "title.textContent=currentLang==='zh'?'训练历史':'Training history'" in PAGE
    assert "title.textContent=currentLang==='zh'?'Code 对话历史':'Code conversations'" in PAGE
    assert 'id="toggleCodeCustomModel"' in PAGE
    assert "['gpt-5.4','GPT-5.4 · 高能力']" in PAGE
    assert "['claude-sonnet-5','Claude Sonnet 5 · 速度与能力']" in PAGE
    assert "['deepseek-v4-flash','DeepSeek V4 Flash · 快速']" in PAGE
    assert "function selectedCodeApiModel()" in PAGE
    assert 'id="codeModelLibrary"' in PAGE
    assert 'id="openApiEditor"' in PAGE
    assert "function wireCodeModelSorting()" in PAGE
    assert "function setCodeDefaultModel(key)" in PAGE
    assert "编辑本地模型" in PAGE
    assert "model-row-menu" in PAGE
    assert "function openUnifiedModelLibrary()" in PAGE
    assert "if(unifiedModelsReturnMode==='training'){showPage('models');setOrbitWorkspaceMode('training');return}" in PAGE
    assert "setOrbitWorkspaceMode(unifiedModelsReturnMode)" in PAGE
    assert "createFreshLaunchState" in PAGE
    assert "revertCodeChanges" in PAGE
    assert "确认撤销这次 Orbit Code" in PAGE
    assert ".code-change-actions{" in PAGE
    assert ".toolbar-nav{gap:6px;padding:0;border:0;background:transparent" in PAGE
    assert ".product-letter,.product-mark{display:block!important" in PAGE
    assert 'font-family:"PingFang SC"' in PAGE
    assert '.product-letter{display:block' in PAGE
    assert '#codeApi.page.active{height:100%;min-height:0;overflow:hidden' in PAGE


def test_local_model_display_name_can_be_edited_without_renaming_checkpoint(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    checkpoint = runtime.models_root / "stable-id.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    result = runtime.rename_model("stable-id", "我的本地模型")
    assert result["name"] == "我的本地模型"
    assert checkpoint.exists()
    assert runtime.list_models()[0]["name"] == "我的本地模型"


def test_external_base_models_are_not_allowed_for_from_scratch_training(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    recommendation = runtime.training_recommendation({
        "preset": "300m", "training_mode": "pretraining", "training_round": 1,
        "base_model_source": "download", "device": "cpu",
    })
    assert recommendation["mode_valid"] is False
    assert any(item["code"] == "external_pretrain_forbidden" for item in recommendation["training_advice"]["items"])
    try:
        runtime._training_mode({
            "training_mode": "pretraining", "training_round": 1,
            "base_model_source": "download",
        })
    except ValueError as exc:
        assert "不能从零预训练" in str(exc)
    else:
        raise AssertionError("external base model unexpectedly passed from-scratch validation")


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
        archive_request = urllib.request.Request(
            base + "/api/conversations/archive",
            data=json.dumps({"id": conversation["id"]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(archive_request) as response:
            assert json.loads(response.read())["status"] == "archived"
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
            entry = teacher["profiles"]["custom"]["entries"][0]
            assert entry["has_api_key"] is True
            assert "api_key" not in entry
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


def test_pretraining_rounds_preserve_parent_lineage_and_corpus_policy(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    payload = {
        "preset": "local", "steps": 1, "batch_size": 1, "seq_len": 8,
        "grad_accum": 1, "learning_rate": 3e-4, "warmup_steps": 0,
        "checkpoint_every": 0, "device": "cpu",
        "training_mode": "pretraining", "training_round": 1,
        "model_name": "round-one", "text": "documents and code. " * 20,
    }
    for name, round_number, parent in [("round-one", 1, ""), ("round-two", 2, "round-one"), ("round-three", 3, "round-two")]:
        runtime.start_training({**payload, "model_name": name, "training_round": round_number, "base_model": parent})
        deadline = time.time() + 20
        while runtime.training_state()["status"] in {"preparing", "running", "stopping"} and time.time() < deadline:
            time.sleep(0.05)
        assert runtime.training_state()["status"] == "completed"
    rows = {row["id"]: row for row in runtime.list_models()}
    assert rows["round-one"]["training_round"] == 1
    assert rows["round-two"]["training_round"] == 2
    assert rows["round-two"]["parent_model"] == "round-one"
    assert rows["round-three"]["training_round"] == 3
    assert rows["round-three"]["parent_model"] == "round-two"
    assert rows["round-three"]["corpus_policy"]["dialogue_ratio"] == 0.05


def test_training_recommendation_exposes_goal_and_task_mix(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    quality = runtime.training_recommendation({
        "preset": "300m", "training_mode": "fine_tuning", "training_round": 1,
        "base_model": "parent", "optimization_goal": "quality", "assisted": True,
    })
    policy = quality["training_advice"]["corpus_policy"]
    assert quality["optimization_goal"] == "quality"
    assert policy["dialogue_ratio"] is None
    assert policy["name"] == "adaptive_quality_diverse_instruction_mix"
    assert "代码/SQL 生成" in policy["task_types"]


def test_training_token_budget_scales_with_model_and_accepts_custom_planning_count(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    small = runtime.training_recommendation({"preset": "300m", "training_mode": "pretraining"})
    large = runtime.training_recommendation({"preset": "38b", "training_mode": "pretraining"})
    assert small["token_planning"]["recommended_tokens"] == 6_000_000_000
    assert large["token_planning"]["recommended_tokens"] == 760_000_000_000
    assert large["recommended_manual_tokens"] > small["recommended_manual_tokens"]

    custom = runtime.training_recommendation({
        "preset": "300m",
        "training_mode": "pretraining",
        "planning_parameter_count": "300,000,000,000,000",
    })
    assert custom["token_planning"]["is_calculator_only"] is True
    assert custom["token_planning"]["target_tokens"] == 6_000_000_000_000_000
    assert custom["token_planning"]["tokens_per_parameter"] == 20


def test_training_token_result_uses_edited_steps_and_advanced_parameters(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    recommendation = runtime.training_recommendation({
        "preset": "300m",
        "training_mode": "pretraining",
        "optimization_goal": "balanced",
        "steps": 2_000,
        "seq_len": 512,
        "batch_size": 1,
        "grad_accum": 1,
        "use_recommended_examples": False,
    })
    planning = recommendation["token_planning"]
    assert planning["final_calculated_tokens"] == 1_024_000
    assert planning["final_formula"] == "2,000 × 512 × 1 × 1 = 1,024,000"
    assert planning["scale_standard_tokens"] == 6_000_000_000
    assert planning["final_deviation_percent"] < -99.9
    assert recommendation["training_advice"]["dataset_estimate"]["optimizer_token_coverage"] == 1_024_000


def test_teacher_api_settings_persist_locally(tmp_path):
    runtime = OrbitRuntime(tmp_path)
    runtime.save_teacher_profile("deepseek", "https://api.deepseek.com", "deepseek-chat", "deep-secret")
    runtime.save_teacher_profile("custom", "https://example.com/v1", "teacher", "custom-secret")
    runtime.save_teacher_profile("deepseek", "https://api.deepseek.com", "deepseek-reasoner", "new-deep-secret")
    reloaded = OrbitRuntime(tmp_path)
    settings = reloaded.teacher_settings()
    assert settings["active_provider"] == "deepseek"
    deepseek = settings["profiles"]["deepseek"]
    custom = settings["profiles"]["custom"]
    assert len(deepseek) == 1
    assert deepseek[0]["model"] == "deepseek-reasoner"
    assert deepseek[0]["api_key"] == "new-deep-secret"
    assert custom[0]["api_key"] == "custom-secret"
    assert reloaded.teacher_settings_path.stat().st_mode & 0o077 == 0


def test_legacy_teacher_api_settings_are_migrated(tmp_path):
    path = tmp_path / "teacher-api.json"
    path.write_text(json.dumps({"base_url": "https://old.example/v1", "model": "old", "api_key": "kept"}))
    settings = OrbitRuntime(tmp_path).teacher_settings()
    assert settings["profiles"]["deepseek"][0]["base_url"] == "https://old.example/v1"
    assert settings["profiles"]["deepseek"][0]["model"] == "old"
    assert settings["profiles"]["deepseek"][0]["api_key"] == "kept"


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


def test_partial_teacher_corpus_survives_failure_and_restart(tmp_path, monkeypatch):
    def fail_after_one_batch(config, api_key, stop_event, callback=None, chunk_callback=None):
        chunk = "标题：基础认知\n这是教师 AI 已经完成并必须保留的第一批训练内容。"
        assert chunk_callback is not None
        chunk_callback(chunk, 5, config.examples)
        if callback:
            callback(5, config.examples)
        raise RuntimeError("教师 API 返回 HTTP 402：Insufficient Balance")

    monkeypatch.setattr(runtime_module, "generate_dataset", fail_after_one_batch)
    runtime = OrbitRuntime(tmp_path)
    runtime.start_auto_training({
        "preset": "300m", "steps": 10, "batch_size": 1, "seq_len": 32,
        "grad_accum": 1, "learning_rate": 3e-4, "warmup_steps": 0,
        "checkpoint_every": 0, "device": "cpu", "model_name": "partial-test",
        "training_mode": "pretraining", "training_round": 1,
        "teacher_provider": "deepseek", "teacher_base_url": "https://api.deepseek.com",
        "teacher_model": "deepseek-chat", "api_key": "sk-test",
        "instruction": "训练基础认知和自然对话能力", "examples": 400,
        "language": "zh", "acknowledge_cost": True,
    })
    deadline = time.time() + 5
    while runtime.training_state()["status"] == "generating" and time.time() < deadline:
        time.sleep(0.02)
    failed = runtime.training_state()
    assert failed["status"] == "failed"
    assert failed["step"] == 5
    assert failed["steps"] == 400
    assert failed["generated_content_available"] is True
    assert "已完成 5/400" in failed["message"]
    assert "第一批训练内容" in runtime.generated_training_content()["content"]

    restored = OrbitRuntime(tmp_path)
    state = restored.training_state()
    assert state["status"] == "failed"
    assert state["step"] == 5
    assert state["steps"] == 400
    assert "第一批训练内容" in restored.generated_training_content()["content"]


def test_code_api_presets_cover_current_chinese_model_providers():
    from orbit.web_ui import PAGE

    for provider in ("Kimi 月之暗面", "智谱 GLM", "阿里云百炼 / Qwen", "MiniMax"):
        assert provider in PAGE
    for model in ("kimi-k3", "kimi-k2.7-code", "glm-5.1", "qwen3-coder-next", "MiniMax-M3"):
        assert model in PAGE
    assert "豆包 / 火山方舟" not in PAGE
    assert "百度千帆 / ERNIE" not in PAGE
    assert "自定义 OpenAI 兼容 API" in PAGE
