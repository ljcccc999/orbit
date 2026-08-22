from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    from .process_name import set_orbit_process_name
    set_orbit_process_name("Orbit Model")
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
        requested_device = os.environ.get("ORBIT_INFERENCE_DEVICE", "auto").strip().lower()
        if requested_device in {"cpu", "mps", "cuda"}:
            if requested_device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("请求使用 MPS，但当前系统不可用")
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("请求使用 CUDA，但当前系统不可用")
            device = requested_device
        elif torch.backends.mps.is_available():
            # On Apple Silicon the byte-level Orbit model is materially faster
            # on MPS than CPU.  Auto must follow the same accelerator choice as
            # local training; forcing CPU made the first chat exceed the
            # 45-second request guard even though MPS was available.
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = model.to(device).eval()
        # MPS lazily compiles kernels on the first forward pass. Without a
        # warm-up, the first real message looks stuck in the thinking state.
        # Finish this cold-start work before reporting the model as ready.
        send({"type": "progress", "progress": 92, "message": "正在预热本地推理引擎"})
        warmup_tokens = min(16, max(1, cfg.max_seq_len))
        warmup = torch.zeros((1, warmup_tokens), dtype=torch.long, device=device)
        with torch.no_grad():
            model(warmup)
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
                # This runtime currently generates without a KV cache. Keep
                # the runtime prefix limited to user-approved memory so every
                # generated token does not recompute a large prefix. Orbit's
                # identity is learned from ORBIT_TRAINING_ANCHOR in training;
                # it is deliberately not answered by a runtime shortcut.
                memory_context = str(request.get("memory_context", "")).strip()[:512]
                user_bytes = f"<|user|>{prompt}\n<|assistant|>".encode("utf-8")
                memory_bytes = f"<|memory|>{memory_context}\n".encode("utf-8") if memory_context else b""
                encoded = (memory_bytes + user_bytes)[-cfg.max_seq_len:]
                ids = torch.tensor([list(encoded)], dtype=torch.long, device=device)
                result = model.generate(
                    ids, max_new_tokens=int(request.get("max_tokens", 128)),
                    temperature=float(request.get("temperature", 0.8)),
                    # Orbit's current training pipeline encodes UTF-8 bytes,
                    # even for the larger architectural vocabulary.
                    vocab_limit=256,
                )
                # A legacy checkpoint may still expose a wider output head.
                # The active Orbit pipeline is byte-level, so never let an
                # out-of-range legacy token crash the API/UI. Clamp it to the
                # byte vocabulary and mark invalid UTF-8 as replacement text.
                generated_ids = [max(0, min(255, int(value))) for value in result[0, ids.shape[1] :].tolist()]
                generated = bytes(generated_ids).decode("utf-8", errors="replace")
                send({"type": "result", "content": generated})
            except Exception as exc:
                send({"type": "error", "error": str(exc)})
    except Exception as exc:
        send({"type": "fatal", "error": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
