import json
import subprocess
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
    assert '<label>智能 <span id="reasoningValue">标准</span>' in PAGE
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
