import plistlib

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
