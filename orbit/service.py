from __future__ import annotations

import json
import os
import platform
import plistlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from xml.sax.saxutils import escape


LABEL = "top.orbit.local"
HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}"


def data_root() -> Path:
    return Path(os.environ.get("ORBIT_DATA_DIR", Path.home() / ".orbit")).expanduser().resolve()


def health() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=1) as response:
            return response.status == 200 and json.loads(response.read()).get("name") == "orbit"
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _mac_plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _mac_domain() -> str:
    return f"gui/{os.getuid()}"


def _mac_install(start: bool) -> Path:
    root = data_root()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = _mac_plist()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable, "-m", "orbit.web", "--no-browser", "--host", HOST,
            "--port", str(PORT), "--data-dir", str(root),
        ],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 5,
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "service.log"),
        "StandardErrorPath": str(logs / "service-error.log"),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }
    temporary = path.with_suffix(".plist.tmp")
    temporary.write_bytes(plistlib.dumps(payload, sort_keys=False))
    os.replace(temporary, path)
    if start:
        _run(["launchctl", "bootout", _mac_domain(), str(path)], check=False)
        _run(["launchctl", "bootstrap", _mac_domain(), str(path)])
        _run(["launchctl", "enable", f"{_mac_domain()}/{LABEL}"], check=False)
    return path


def _linux_unit() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "orbit.service"


def _linux_install(start: bool) -> Path:
    root = data_root()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = _linux_unit()
    path.parent.mkdir(parents=True, exist_ok=True)
    unit = f"""[Unit]
Description=Orbit local AI API
After=network.target

[Service]
Type=simple
ExecStart={sys.executable} -m orbit.web --no-browser --host {HOST} --port {PORT} --data-dir {root}
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(unit, encoding="utf-8")
    os.replace(temporary, path)
    _run(["systemctl", "--user", "daemon-reload"])
    if start:
        _run(["systemctl", "--user", "enable", "--now", "orbit.service"])
    return path


def _windows_task_xml() -> Path:
    return data_root() / "service" / "orbit-task.xml"


def _windows_install(start: bool) -> Path:
    root = data_root()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = _windows_task_xml()
    path.parent.mkdir(parents=True, exist_ok=True)
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    executable = pythonw if pythonw.exists() else Path(sys.executable)
    arguments = f'-m orbit.web --no-browser --host {HOST} --port {PORT} --data-dir "{root}"'
    xml = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>true</StartWhenAvailable><RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure><ExecutionTimeLimit>PT0S</ExecutionTimeLimit><Enabled>true</Enabled></Settings>
  <Actions Context="Author"><Exec><Command>{escape(str(executable))}</Command><Arguments>{escape(arguments)}</Arguments></Exec></Actions>
</Task>'''
    path.write_text(xml, encoding="utf-16")
    _run(["schtasks", "/Create", "/TN", "OrbitLocalAI", "/XML", str(path), "/F"])
    if start:
        _run(["schtasks", "/Run", "/TN", "OrbitLocalAI"])
    return path


def install(start: bool = True) -> Path:
    system = platform.system()
    if system == "Darwin":
        return _mac_install(start)
    if system == "Linux":
        return _linux_install(start)
    if system == "Windows":
        return _windows_install(start)
    raise RuntimeError("后台服务目前支持 macOS、Linux 和 Windows")


def start() -> None:
    system = platform.system()
    if system == "Darwin":
        path = _mac_plist()
        if not path.exists():
            install(start=True)
            return
        result = _run(["launchctl", "kickstart", "-k", f"{_mac_domain()}/{LABEL}"], check=False)
        if result.returncode != 0:
            _run(["launchctl", "bootstrap", _mac_domain(), str(path)])
    elif system == "Linux":
        if not _linux_unit().exists():
            install(start=True)
            return
        _run(["systemctl", "--user", "start", "orbit.service"])
    elif system == "Windows":
        result = _run(["schtasks", "/Run", "/TN", "OrbitLocalAI"], check=False)
        if result.returncode != 0:
            install(start=True)
    else:
        raise RuntimeError("后台服务目前支持 macOS 和 Linux")


def set_autostart(enabled: bool) -> None:
    """Toggle login-time service startup without stopping the live agent."""
    system = platform.system()
    if system == "Darwin":
        path = _mac_plist()
        if enabled and not path.exists():
            install(start=False)
        _run(["launchctl", "enable" if enabled else "disable", f"{_mac_domain()}/{LABEL}"], check=False)
    elif system == "Linux":
        if enabled and not _linux_unit().exists():
            install(start=False)
        _run(["systemctl", "--user", "enable" if enabled else "disable", "orbit.service"], check=False)
    elif system == "Windows":
        if enabled and not _windows_task_xml().exists():
            install(start=False)
        _run(["schtasks", "/Change", "/TN", "OrbitLocalAI", "/ENABLE" if enabled else "/DISABLE"], check=False)


def stop() -> None:
    if platform.system() == "Darwin":
        _run(["launchctl", "bootout", _mac_domain(), str(_mac_plist())], check=False)
    elif platform.system() == "Linux":
        _run(["systemctl", "--user", "stop", "orbit.service"], check=False)
    elif platform.system() == "Windows":
        _run(["schtasks", "/End", "/TN", "OrbitLocalAI"], check=False)


def uninstall() -> None:
    if platform.system() == "Darwin":
        path = _mac_plist()
        _run(["launchctl", "bootout", _mac_domain(), str(path)], check=False)
        path.unlink(missing_ok=True)
    elif platform.system() == "Linux":
        _run(["systemctl", "--user", "disable", "--now", "orbit.service"], check=False)
        _linux_unit().unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"], check=False)
    elif platform.system() == "Windows":
        _run(["schtasks", "/End", "/TN", "OrbitLocalAI"], check=False)
        _run(["schtasks", "/Delete", "/TN", "OrbitLocalAI", "/F"], check=False)
        _windows_task_xml().unlink(missing_ok=True)


def ensure_running(timeout: float = 20) -> None:
    if health():
        return
    install(start=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health():
            return
        time.sleep(0.25)
    raise RuntimeError(f"Orbit 后台服务没有在 {URL} 就绪；请运行 orbit logs 查看日志")


def log_paths() -> tuple[Path, Path]:
    root = data_root() / "logs"
    return root / "service.log", root / "service-error.log"
