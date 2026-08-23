from pathlib import Path
import tomllib

from orbit import updater


def test_checkout_version_wins_over_stale_installed_metadata():
    manifest = Path(updater.__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]["version"]
    assert updater.current_version() == expected


def test_reload_background_service_forces_fresh_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr("orbit.service.stop", lambda: calls.append("stop"))
    monkeypatch.setattr("orbit.service.start", lambda: calls.append("start"))
    monkeypatch.setattr("orbit.service.ensure_running", lambda timeout: calls.append(("healthy", timeout)))
    monkeypatch.setattr(updater.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))

    updater._reload_background_service()

    assert calls == ["stop", ("sleep", 0.5), "start", ("healthy", 30)]
