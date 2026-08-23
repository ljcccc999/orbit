from pathlib import Path
import tomllib

from orbit import updater


def test_checkout_version_wins_over_stale_installed_metadata():
    manifest = Path(updater.__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]["version"]
    assert updater.current_version() == expected
