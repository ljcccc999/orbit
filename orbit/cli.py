from __future__ import annotations

import argparse
import json
import urllib.request
import webbrowser

from . import service
from . import updater


def _post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        service.URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbit local AI service")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("open", help="start Orbit and open the web interface")
    commands.add_parser("start", help="start the background API")
    commands.add_parser("stop", help="stop the background API")
    commands.add_parser("restart", help="restart the background API")
    commands.add_parser("status", help="show background API status")
    commands.add_parser("logs", help="show log file locations")
    commands.add_parser("unload", help="release the active model from memory")
    update = commands.add_parser("update", help="check for and install the latest Orbit runtime")
    update.add_argument("--check", action="store_true", help="only check for a newer release")
    chat = commands.add_parser("chat", help="send one message without opening the web interface")
    chat.add_argument("message")
    chat.add_argument("--model")
    service_command = commands.add_parser("service", help="install or remove automatic startup")
    service_command.add_argument("action", choices=["install", "uninstall"])
    args = parser.parse_args()

    if args.command in {None, "open"}:
        service.ensure_running()
        webbrowser.open(service.URL)
        print(f"Orbit：{service.URL}")
    elif args.command == "start":
        service.ensure_running()
        print(f"Orbit API 已在后台运行：{service.URL}/v1")
    elif args.command == "stop":
        service.stop()
        print("Orbit 后台服务已停止")
    elif args.command == "restart":
        service.stop()
        service.ensure_running()
        print(f"Orbit 已重新启动：{service.URL}/v1")
    elif args.command == "status":
        print("running" if service.health() else "stopped")
    elif args.command == "logs":
        print("\n".join(str(path) for path in service.log_paths()))
    elif args.command == "unload":
        service.ensure_running()
        print(json.dumps(_post("/api/models/unload", {}), ensure_ascii=False))
    elif args.command == "update":
        info = updater.check()
        print(json.dumps(updater.as_dict(info), ensure_ascii=False))
        if info.error or args.check or not info.available:
            return
        print("Orbit 更新已排队；如果正在训练，会等 checkpoint 保存后再安全重启。")
        if not updater.schedule_install(info):
            raise SystemExit("无法排队 Orbit 更新")
    elif args.command == "chat":
        service.ensure_running()
        payload = {"prompt": args.message}
        if args.model:
            payload["model"] = args.model
        print(_post("/api/chat", payload)["content"])
    elif args.command == "service":
        if args.action == "install":
            path = service.install(start=True)
            print(f"后台服务已安装：{path}")
        else:
            service.uninstall()
            print("后台服务已移除；模型和训练数据未删除")
