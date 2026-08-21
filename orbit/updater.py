from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path


REPOSITORY = "ljcccc999/orbit"
RELEASES_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"


def _download_environment() -> dict[str, str]:
    """Give the updater a local proxy only when one is actually listening.

    LaunchAgents do not inherit the interactive shell's proxy variables on
    macOS. The probe is local-only and does not change system proxy settings.
    """
    env = os.environ.copy()
    if any(env.get(key) for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")):
        return env
    try:
        with socket.create_connection(("127.0.0.1", 7890), timeout=0.2):
            pass
    except OSError:
        return env
    proxy = "http://127.0.0.1:7890"
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        env[key] = proxy
    env.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    env.setdefault("no_proxy", env["NO_PROXY"])
    return env


def _urlopen(request: urllib.request.Request, timeout: float):
    env = _download_environment()
    proxy = env.get("HTTPS_PROXY") or env.get("https_proxy") or env.get("ALL_PROXY") or env.get("all_proxy")
    if not proxy:
        return urllib.request.urlopen(request, timeout=timeout)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return opener.open(request, timeout=timeout)


def current_version() -> str:
    try:
        return metadata.version("orbit-ai")
    except metadata.PackageNotFoundError:
        return "0.6.9"


def _version_tuple(value: str) -> tuple[int, ...]:
    raw = value.strip().lstrip("vV").split("+")[0]
    parts: list[int] = []
    for part in raw.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits or "0"))
    return tuple((parts + [0, 0, 0])[:3])


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    latest_version: str | None
    available: bool
    release_url: str | None = None
    tag: str | None = None
    error: str | None = None


def check() -> UpdateInfo:
    request = urllib.request.Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Orbit-Updater"},
    )
    version = current_version()
    try:
        with _urlopen(request, timeout=8) as response:
            payload = json.loads(response.read())
        tag = str(payload.get("tag_name", ""))
        latest = tag.lstrip("vV")
        if not latest:
            raise ValueError("GitHub returned a release without a version")
        return UpdateInfo(
            current_version=version,
            latest_version=latest,
            available=_version_tuple(latest) > _version_tuple(version),
            release_url=str(payload.get("html_url") or "") or None,
            tag=tag,
        )
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return UpdateInfo(version, None, False, error=f"update check failed: {exc}")


def _download_script(name: str, tag: str) -> Path:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{tag}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "Orbit-Updater"})
    with _urlopen(request, timeout=20) as response:
        content = response.read()
    folder = Path(tempfile.mkdtemp(prefix="orbit-update-"))
    path = folder / name
    path.write_bytes(content)
    return path


def install_latest(info: UpdateInfo) -> int:
    if not info.available or not info.tag:
        return 0
    archive = f"https://github.com/{REPOSITORY}/archive/refs/tags/{info.tag}.tar.gz"
    env = _download_environment()
    env.update({"ORBIT_ARCHIVE_URL": archive, "ORBIT_NO_BROWSER": "1"})
    if platform.system() == "Windows":
        script = _download_script("install.ps1", info.tag)
        return subprocess.call(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], env=env)
    script = _download_script("install.sh", info.tag)
    return subprocess.call(["/bin/sh", str(script)], env=env)


def schedule_install(info: UpdateInfo) -> bool:
    """Start an updater outside the service process before stopping it."""
    if not info.available or not info.tag:
        return False
    code = f"""
import json
import time
import urllib.request
time.sleep(1)
active = {{'preparing', 'generating', 'waiting_memory', 'needs_memory', 'running', 'stopping'}}
errors = 0
while True:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/api/training', timeout=3) as response:
            status = json.loads(response.read()).get('status')
        errors = 0
        if status not in active:
            break
    except Exception:
        errors += 1
        if errors >= 12:
            break
    time.sleep(5)
from orbit import service, updater
service.stop()
raise SystemExit(updater.install_latest(updater.UpdateInfo(
    {info.current_version!r}, {info.latest_version!r}, True,
    {info.release_url!r}, {info.tag!r}, None
)))
"""
    subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    return True


def as_dict(info: UpdateInfo) -> dict[str, object]:
    return asdict(info)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        print(json.dumps(as_dict(check()), ensure_ascii=False))
        raise SystemExit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "--install-latest":
        info = check()
        raise SystemExit(install_latest(info))
    raise SystemExit("usage: python -m orbit.updater --check")
