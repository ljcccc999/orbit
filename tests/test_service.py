import plistlib
import subprocess

from orbit import service


def test_macos_service_restarts_after_failure_without_opening_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ORBIT_DATA_DIR", str(tmp_path / ".orbit"))
    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")
    path = service.install(start=False)
    payload = plistlib.loads(path.read_bytes())
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert "--no-browser" in payload["ProgramArguments"]


def test_macos_autostart_toggle_does_not_stop_live_service(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")
    calls = []

    def fake_run(command, check=True):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(service, "_run", fake_run)
    service.set_autostart(False)

    assert calls == [["launchctl", "disable", f"gui/{service.os.getuid()}/{service.LABEL}"]]
    assert all("bootout" not in command for command in calls)
