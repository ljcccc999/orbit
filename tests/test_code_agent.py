import hashlib
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import pytest

from orbit.code_agent import OrbitCodeAgent
from orbit.web_ui import PAGE


def _agent(tmp_path: Path) -> OrbitCodeAgent:
    return OrbitCodeAgent(tmp_path, lambda *args: {"content": "{}"}, lambda: [{"id": "orbit-local", "name": "Orbit Local"}])


def test_code_api_profiles_are_multiple_and_keys_stay_private(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = _agent(tmp_path)
    first = agent.save_settings({
        "workspace": str(workspace), "provider": "api", "save_profile": True,
        "profile_name": "Coding A", "base_url": "https://api.example.com/v1",
        "model": "model-a", "api_key": "secret-a",
    })
    second = agent.save_settings({
        "workspace": str(workspace), "provider": "api", "save_profile": True,
        "profile_name": "Coding B", "base_url": "https://other.example.com/v1",
        "model": "model-b", "api_key": "secret-b",
    })

    assert len(second["profiles"]) == 2
    assert all("api_key" not in row for row in second["profiles"])
    assert {row["key_hint"] for row in second["profiles"]} == {"••••et-a", "••••et-b"}
    stored = json.loads(agent.settings_path.read_text(encoding="utf-8"))
    assert {row["api_key"] for row in stored["profiles"]} == {"secret-a", "secret-b"}
    agent.delete_profile(first["profiles"][0]["id"])
    assert len(agent.public_settings()["profiles"]) == 1


def test_code_model_order_is_persisted_for_api_and_local_models(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = _agent(tmp_path)
    saved = agent.save_settings({
        "workspace": str(workspace), "provider": "api", "save_profile": True,
        "profile_name": "Cloud", "base_url": "https://api.example.com/v1",
        "model": "cloud-model", "api_key": "secret",
    })
    api_key = "api:" + saved["profiles"][0]["id"]
    updated = agent.save_settings({
        "workspace": str(workspace), "provider": "local", "model": "orbit-local",
        "model_order": ["local:orbit-local", api_key],
    })
    assert updated["model_order"] == ["local:orbit-local", api_key]
    assert updated["provider"] == "local"
    assert updated["model"] == "orbit-local"


def test_code_creates_a_default_orbit_workspace_and_uses_shared_local_default(tmp_path):
    calls = []

    def local_chat(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return {"model": kwargs.get("model_id"), "content": "ok"}

    agent = OrbitCodeAgent(tmp_path, local_chat, lambda: [{"id": "orbit-local", "name": "Orbit Local"}])
    settings = agent.public_settings()
    assert Path(settings["workspace"]).is_dir()
    assert Path(settings["workspace"]).name == "workspace"
    result = agent.chat_default("hello", max_tokens=16)
    assert result["content"] == "ok"
    assert calls == [("hello", {"model_id": "orbit-local", "max_tokens": 16, "temperature": 0.8})]


def test_permissions_keep_web_search_separate_from_shell_network_and_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    agent = _agent(tmp_path / "code")

    assert agent._action_may_leave_workspace({"tool": "web_search", "query": "Orbit"}, workspace) is False
    assert agent._action_may_leave_workspace({"tool": "shell", "command": "curl https://example.com"}, workspace) is True
    assert agent._action_may_leave_workspace(
        {"tool": "shell", "command": f"/bin/sh -c 'printf hello > {workspace / 'new.txt'}'"},
        workspace,
    ) is False
    assert agent._action_may_leave_workspace(
        {"tool": "shell", "command": f"printf hello > {outside}"},
        workspace,
    ) is True
    assert agent._action_may_leave_workspace(
        {"tool": "shell", "command": f"/bin/sh -c 'printf hello > {outside}'"},
        workspace,
    ) is True
    assert agent._action_may_leave_workspace({"tool": "read_file", "path": str(outside)}, workspace) is True
    assert agent._action_may_leave_workspace({"tool": "read_file", "path": "README.md"}, workspace) is False
    assert agent._action_may_leave_workspace(
        {"tool": "apply_patch", "patch": "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+hello"},
        workspace,
    ) is False
    assert agent._action_may_leave_workspace(
        {"tool": "apply_patch", "patch": f"--- /dev/null\n+++ {outside}\n@@ -0,0 +1 @@\n+hello"},
        workspace,
    ) is True
    assert agent._action_is_high_risk({"tool": "shell", "command": "rm -rf build"}) is True
    assert agent._action_is_high_risk({"tool": "apply_patch", "patch": "--- /dev/null\n+++ b/new.txt"}) is False


def test_approval_wakes_waiting_action_and_execution_continues(monkeypatch, tmp_path):
    agent = _agent(tmp_path / "code")
    session_id = "e" * 24
    row = {
        "id": session_id,
        "status": "running",
        "events": [],
        "updated_at": "",
        "pending_approval": None,
    }
    agent._sessions[session_id] = row
    agent._save(row)
    monkeypatch.setattr(agent, "_execute", lambda *_args: "continued after approval")
    result = {}

    worker = threading.Thread(
        target=lambda: result.update(agent._execute_with_policy(
            row,
            {"tool": "shell", "command": "curl https://example.com", "summary": "访问外部网络"},
            {"permission": "ask", "computer_control": False},
            tmp_path,
            threading.Event(),
        )),
        daemon=True,
    )
    worker.start()
    deadline = time.monotonic() + 2
    while row["pending_approval"] is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert row["status"] == "waiting_approval"
    approved = agent.approve(session_id, True)
    assert approved["status"] == "running"
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result["ok"] is True
    assert result["output"] == "continued after approval"
    assert row["pending_approval"] is None
    assert row["events"][-1]["status"] == "completed"


def test_approval_restarts_a_persisted_wait_without_a_live_worker(monkeypatch, tmp_path):
    agent = _agent(tmp_path / "code")
    session_id = "a" * 24
    row = {
        "id": session_id, "title": "resume", "prompt": "continue", "status": "waiting_approval",
        "events": [], "updated_at": "", "settings": {"workspace": str(agent.workspace_root)},
        "pending_approval": {"id": "old", "summary": "访问外部文件", "tool": "read_file", "decision": None},
    }
    agent._sessions[session_id] = row
    agent._save(row)
    resumed = threading.Event()

    def fake_run(restored, settings, stop, run_gate, baseline):
        assert restored["resume_approval"] == {"tool": "read_file", "summary": "访问外部文件"}
        assert settings["workspace"] == str(agent.workspace_root)
        assert run_gate.is_set()
        resumed.set()

    monkeypatch.setattr(agent, "_run", fake_run)
    approved = agent.approve(session_id, True)

    assert approved["status"] == "planning"
    assert approved["pending_approval"] is None
    assert resumed.wait(2)


def test_code_pause_and_resume_use_the_same_safe_run_gate(tmp_path):
    agent = _agent(tmp_path / "code")
    session_id = "a" * 24
    row = {"id": session_id, "status": "running", "events": [], "updated_at": ""}
    gate = threading.Event()
    gate.set()
    agent._sessions[session_id] = row
    agent._run_gates[session_id] = gate
    agent._save(row)

    paused = agent.toggle_pause(session_id)
    assert paused["status"] == "paused"
    assert not gate.is_set()
    assert paused["events"][-1]["title"] == "已暂停"

    resumed = agent.toggle_pause(session_id)
    assert resumed["status"] == "running"
    assert gate.is_set()
    assert resumed["events"][-1]["title"] == "已继续运行"


def test_live_guidance_answers_before_queue_marker(tmp_path):
    agent = _agent(tmp_path / "code")
    session_id = "a" * 24
    row = {"id": session_id, "status": "running", "events": [], "updated_at": "", "settings": {}}
    agent._sessions[session_id] = row
    agent._save(row)

    result = agent.guide(session_id, "请先说明当前发现", "steer")
    events = result["events"]
    assert [event["kind"] for event in events[-2:]] == ["assistant", "guidance"]
    assert events[-2]["phase"] == "guidance_reply"
    assert "先回答" in events[-2]["title"]
    assert events[-1]["mode"] == "steer"


def test_revert_changes_restores_only_the_archived_session_state(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    target = workspace / "hello.txt"
    target.write_text("before\n", encoding="utf-8")
    agent = _agent(tmp_path / "code")
    session_id = "f" * 24
    before = target.read_bytes()
    target.write_text("after\n", encoding="utf-8")
    after = target.read_bytes()
    row = {
        "id": session_id, "status": "completed", "settings": {"workspace": str(workspace)},
        "changes": {"files": [{"path": "hello.txt", "status": "modified", "before_sha256": hashlib.sha256(before).hexdigest(), "after_sha256": hashlib.sha256(after).hexdigest()}]},
        "events": [], "updated_at": "",
    }
    agent._sessions[session_id] = row
    agent._archive_review_files(row, {"hello.txt": before}, workspace)
    agent._save(row)
    restored = agent.revert_changes(session_id)
    assert target.read_text(encoding="utf-8") == "before\n"
    assert restored["changes_reverted"] is True


def test_promoted_guidance_cannot_be_cancelled(tmp_path):
    agent = _agent(tmp_path)
    session_id = "a" * 24
    row = {
        "id": session_id, "status": "running", "events": [], "updated_at": "",
        "directives": [{"id": "d1", "mode": "queue", "prompt": "排队消息", "consumed": False, "deleted": False}],
    }
    agent._sessions[session_id] = row
    agent._save(row)

    promoted = agent.update_guidance(session_id, "d1", "steer")
    assert promoted["directives"][0]["mode"] == "steer"
    with pytest.raises(ValueError, match="不可撤销"):
        agent.update_guidance(session_id, "d1", "delete")


def test_file_review_is_relative_to_session_baseline(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    existing = workspace / "existing.txt"
    existing.write_text("user change before agent\n", encoding="utf-8")
    baseline = OrbitCodeAgent._snapshot_workspace(workspace)

    existing.write_text("user change before agent\nagent line\n", encoding="utf-8")
    (workspace / "new.py").write_text("print('orbit')\n", encoding="utf-8")
    changes = OrbitCodeAgent._workspace_changes(workspace, baseline)

    assert changes["files_changed"] == 2
    assert changes["additions"] == 2
    assert changes["deletions"] == 0
    assert {row["path"] for row in changes["files"]} == {"existing.txt", "new.py"}


def test_orbit_code_ui_contains_queue_progress_review_and_compact_tools():
    assert 'id="code" class="page"' in PAGE
    assert 'id="queuedGuides"' in PAGE
    assert 'id="codeGuideMode"' not in PAGE
    assert "确认撤销这条排队消息" in PAGE
    assert "action:'steer'" in PAGE
    assert "renderCompactCodeTimeline" in PAGE
    assert "搜索文件/代码" in PAGE
    assert "搜索网页" in PAGE
    assert 'id="currentCodeActivity"' in PAGE
    assert "正在思考" in PAGE
    assert 'id="reviewDrawer"' in PAGE
    assert 'id="stepRing"' in PAGE
    assert 'id="codeLocalContext"' in PAGE
    assert 'id="codeLongMemory"' in PAGE
    assert "session-spinner" in PAGE
    assert "对话历史" in PAGE
    assert "展开执行过程" in PAGE
    assert '<label>智能 <span id="reasoningValue">中</span>' in PAGE
    assert '<span>极高</span><span>最高</span><span>Ultra</span>' in PAGE
    assert "耗时和 token 也更多" in PAGE
    assert "renderThreeLevelCodeTimeline" in PAGE
    assert "code-stage" in PAGE
    assert ".diff-view-line.diff-add" in PAGE
    assert ".diff-view-line.diff-del" in PAGE
    assert 'id="openCodePlugins"' in PAGE
    assert 'id="codePluginDrawer"' in PAGE
    assert 'id="codeApiPreset"' in PAGE
    assert "自定义 OpenAI 兼容 API" in PAGE
    assert "查看整个文件" in PAGE
    assert "side-settings-button" in PAGE
    assert "nav.insertBefore(plugins,models)" in PAGE
    assert "trainingResetFor='',modelNameConflict=false" in PAGE
    assert "}};['steps','batch','seq','accum'].forEach" in PAGE
    assert 'id="preventSleep"' in PAGE
    assert 'id="backgroundService"' in PAGE
    assert 'id="computerControl"' in PAGE
    assert "允许操控鼠标和键盘" in PAGE
    assert "释放本机 API 端口" in PAGE
    assert "formatCodeElapsed" in PAGE
    assert "已耗时 " in PAGE
    assert "if(bar)bar.hidden=true" in PAGE
    assert "event.phase==='summary'" in PAGE
    assert "codeStageActionLabel" in PAGE
    assert "codeRunningActionLabel" in PAGE
    assert "正在查询文件/代码" in PAGE
    assert "正在操作鼠标键盘" in PAGE
    assert "codeToolKind" in PAGE
    assert "tool-output-foot" in PAGE
    assert "code-stage-message" in PAGE
    assert "编辑 ${counts.modified} 个文件" in PAGE
    assert "搜索网页 ${web} 次" in PAGE
    assert "previousScroll" in PAGE
    assert "timeline.scrollTop=wasNearBottom?timeline.scrollHeight:previousScroll" in PAGE
    assert "已批准，Orbit Code 正在继续执行" in PAGE
    assert "if(event.kind==='approval_decision')return ''" in PAGE
    assert "请求批准 · 等待批准" in PAGE
    assert ".approval-box,.code-stage,.tool-run-group,.tool-run-row" in PAGE
    assert ".step-ring:before{display:none}" in PAGE
    assert 'border-radius:999px!important' in PAGE
    assert 'border-radius:50%!important;background:#fff!important' in PAGE
    assert '-webkit-app-region:no-drag!important;cursor:pointer!important' in PAGE
    assert "showPage('chat');newConversation()" in PAGE


def test_long_term_memory_keeps_completed_task_summaries_not_file_bodies(tmp_path):
    agent = _agent(tmp_path)
    row = {
        "id": "b" * 24,
        "title": "修复登录页",
        "status": "completed",
        "updated_at": "2026-08-23T12:00:00+0800",
        "settings": {"workspace": str(tmp_path)},
        "changes": {"files": [{"path": "app.py", "diff": "+SECRET_BODY"}]},
        "events": [{"kind": "assistant", "phase": "summary", "detail": "登录页已经修复并通过测试。"}],
    }
    agent._sessions[row["id"]] = row
    agent._remember(row)

    stored = agent.memory_path.read_text(encoding="utf-8")
    assert "登录页已经修复并通过测试" in stored
    assert "app.py" in stored
    assert "SECRET_BODY" not in stored
    assert "长期记忆" in agent._long_term_context(tmp_path)


def test_intelligence_levels_scale_execution_and_token_budgets(tmp_path):
    agent = _agent(tmp_path)
    instant = agent._intelligence_profile({"reasoning": "none"})
    standard = agent._intelligence_profile({"reasoning": "medium"})
    maximum = agent._intelligence_profile({"reasoning": "max"})

    assert instant["max_turns"] < standard["max_turns"] < maximum["max_turns"]
    assert instant["output_tokens"] < standard["output_tokens"] < maximum["output_tokens"]
    assert maximum["max_actions"] >= standard["max_actions"]
    assert "工作区" not in agent._system_prompt({"reasoning": "max"}, tmp_path)
    assert "只有没有可靠命令行/API 路径" in agent._system_prompt({"reasoning": "max"}, tmp_path)
    assert "你的产品身份始终是 Orbit" in agent._system_prompt({"reasoning": "max"}, tmp_path)
    assert "由 YUNSH 开发" in agent._system_prompt({"reasoning": "max"}, tmp_path)
    assert "阶段说明 → 可展开的工具汇总 → 下一阶段说明" in agent._system_prompt({"reasoning": "max"}, tmp_path)


def test_plugins_are_local_toggleable_and_injected_as_context(tmp_path):
    agent = _agent(tmp_path)
    rows = agent.install_plugin({
        "id": "orbit.docs", "name": "Docs", "version": "1.0.0",
        "description": "Project conventions", "instructions": "Always inspect README.md first.",
    })
    assert rows[0]["enabled"] is False
    rows = agent.toggle_plugin("orbit.docs", True)
    assert rows[0]["enabled"] is True
    assert "Always inspect README.md first" in agent._plugin_context()


def test_full_file_review_uses_archived_before_content_for_deleted_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    deleted = workspace / "old.py"
    deleted.write_text("first\nremoved line\nlast\n", encoding="utf-8")
    baseline = OrbitCodeAgent._snapshot_workspace(workspace)
    deleted.unlink()
    agent = _agent(tmp_path)
    row = {
        "id": "c" * 24, "status": "completed", "settings": {"workspace": str(workspace)},
        "changes": OrbitCodeAgent._workspace_changes(workspace, baseline), "events": [], "updated_at": "",
    }
    agent._sessions[row["id"]] = row
    agent._save(row)
    agent._archive_review_files(row, baseline, workspace)
    viewed = agent.read_session_file(row["id"], "old.py")
    assert viewed["status"] == "deleted"
    assert "removed line" in viewed["content"]


def test_computer_tool_builds_argv_without_shell(monkeypatch, tmp_path):
    agent = _agent(tmp_path)
    captured = {}
    monkeypatch.setattr("orbit.code_agent.shutil.which", lambda name: "/usr/local/bin/cliclick")

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("orbit.code_agent.subprocess.run", fake_run)
    result = agent._computer_action({"action": "hotkey", "keys": ["cmd", "shift"], "key": "p"})
    assert "已执行" in result
    assert captured["argv"][-3:] == ["kd:cmd,shift", "t:p", "ku:cmd,shift"]
    assert captured["kwargs"].get("shell") is None


def test_computer_control_setting_blocks_and_logs_action(monkeypatch, tmp_path):
    agent = _agent(tmp_path)
    row = {"id": "d" * 24, "events": [], "updated_at": "", "pending_approval": None, "status": "running"}
    agent._sessions[row["id"]] = row
    result = agent._execute_with_policy(
        row, {"tool": "computer", "action": "click", "x": 20, "y": 30, "summary": "点击确认按钮"},
        {"permission": "full", "computer_control": False}, tmp_path, __import__("threading").Event(),
    )
    assert result["ok"] is False
    assert "尚未允许" in result["error"]
    assert row["events"][-1]["tool"] == "computer"
    assert row["events"][-1]["status"] == "failed"


def test_computer_control_sync_does_not_revalidate_incomplete_api_profile(tmp_path):
    agent = _agent(tmp_path)
    values = agent._defaults()
    values.update(provider="api", model="", api_key="", computer_control=False)
    agent.settings_path.write_text(json.dumps(values), encoding="utf-8")

    result = agent.set_computer_control(True)

    assert result["computer_control"] is True
    assert result["provider"] == "api"


def test_web_search_results_keep_keyword_content_and_real_destination():
    page = '''
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Orbit docs</a>
    <div class="result__snippet">A <b>searchable</b> result summary.</div>
    '''
    rows = OrbitCodeAgent._parse_web_results(page)
    assert rows == [{
        "title": "Orbit docs",
        "url": "https://example.com/doc",
        "snippet": "A searchable result summary.",
    }]


def test_parse_reply_recovers_wrapped_json_object():
    value = OrbitCodeAgent._parse_reply(
        '<think>brief internal provider wrapper</think>\n'
        '{"phase":"plan","message":"开始","actions":[],"done":false}\n'
    )

    assert value["phase"] == "plan"
    assert value["message"] == "开始"
    assert value["done"] is False


def test_openai_compatible_api_retries_without_optional_fields(monkeypatch, tmp_path):
    agent = _agent(tmp_path)
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 400, "bad request", {}, BytesIO(b"unsupported field"))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = agent._api_model(
        "system", "prompt", [],
        {"api_format": "openai", "base_url": "https://api.example.com/v1", "api_key": "secret", "model": "example", "reasoning": "high"},
        [],
    )

    assert result == "{}"
    assert "response_format" in requests[0]
    assert "reasoning_effort" in requests[0]
    assert "response_format" not in requests[1]
    assert "reasoning_effort" not in requests[1]
