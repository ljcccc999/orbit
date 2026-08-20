from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        send({"type": "progress", "progress": 20, "message": "正在启动隔离推理进程"})
        import torch
        from .config import OrbitConfig
        from .model import OrbitForCausalLM

        send({"type": "progress", "progress": 35, "message": "正在读取本地 checkpoint"})
        try:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(args.checkpoint, map_location="cpu")
        cfg = OrbitConfig(**state["config"])
        send({"type": "progress", "progress": 60, "message": "正在分配模型结构"})
        model = OrbitForCausalLM(cfg)
        send({"type": "progress", "progress": 82, "message": "正在载入模型权重"})
        model.load_state_dict(state["model"])
        del state
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = model.to(device).eval()
        send({"type": "ready", "progress": 100, "message": f"模型已加载到 {device}", "device": device})

        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    send({"type": "stopped"})
                    return
                if request.get("command") != "chat":
                    raise ValueError("unknown command")
                prompt = str(request.get("prompt", ""))
                system_prompt = str(request.get("system_prompt", "")).strip()
                model_name = str(request.get("model_name", "Orbit")).strip() or "Orbit"
                framed_prompt = f"<|system|>{system_prompt} The current model name is {model_name}.\n<|user|>{prompt}\n<|assistant|>"
                encoded = framed_prompt.encode("utf-8")[-cfg.max_seq_len :]
                ids = torch.tensor([list(encoded)], dtype=torch.long, device=device)
                result = model.generate(
                    ids, max_new_tokens=int(request.get("max_tokens", 128)),
                    temperature=float(request.get("temperature", 0.8)),
                )
                generated = bytes(result[0, ids.shape[1] :].tolist()).decode("utf-8", errors="replace")
                send({"type": "result", "content": generated})
            except Exception as exc:
                send({"type": "error", "error": str(exc)})
    except Exception as exc:
        send({"type": "fatal", "error": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
