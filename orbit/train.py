from __future__ import annotations

import argparse
import math
import os
import shutil
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch

from .config import OrbitConfig
from .model import OrbitForCausalLM
from .training_config import TrainingConfig


def byte_batch(text: str, batch_size: int, seq_len: int, device: torch.device):
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    if data.numel() < seq_len + 2:
        raise ValueError("训练文本必须比序列长度更长")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    return x, y


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _save(path: Path, model, optimizer, scheduler, cfg, preset, train_cfg, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    estimated = max(512 * 1024 * 1024, cfg.estimate_parameters() * 16)
    if free < estimated + 1024 * 1024 * 1024:
        raise OSError("磁盘可用空间接近安全下限，Orbit 未覆盖现有 checkpoint")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({
            "config": cfg.__dict__, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "preset": preset, "training_config": train_cfg.__dict__, "step": step,
        }, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(
    steps: int = 100, batch_size: int = 2, seq_len: int = 64,
    learning_rate: float = 3e-4, device_name: str = "auto",
    checkpoint: Path = Path("orbit-checkpoint.pt"), text: str = "",
    preset: str = "local", callback: Optional[Callable[[int, float], None]] = None,
    stop_event: Optional[threading.Event] = None, grad_accum: int = 1,
    warmup_steps: int = 0, weight_decay: float = 0.1, grad_clip: float = 1.0,
    precision: str = "auto", scheduler_name: str = "cosine",
    checkpoint_every: int = 500, seed: int = 42, resume: Optional[Path] = None,
    training_config: Optional[TrainingConfig] = None,
) -> Path:
    train_cfg = training_config or TrainingConfig(
        steps=steps, batch_size=batch_size, seq_len=seq_len, grad_accum=grad_accum,
        learning_rate=learning_rate, warmup_steps=warmup_steps, weight_decay=weight_decay,
        grad_clip=grad_clip, precision=precision, scheduler=scheduler_name,
        checkpoint_every=checkpoint_every, seed=seed,
    )
    train_cfg.validate()
    torch.manual_seed(train_cfg.seed)
    cfg = OrbitConfig.for_preset(preset).with_overrides(max_seq_len=max(train_cfg.seq_len, OrbitConfig.for_preset(preset).max_seq_len))
    device = select_device(device_name)
    if device.type in {"mps", "cpu"}:
        memory = cfg.memory_check()
        if not memory["can_train"]:
            raise MemoryError(
                f"{preset} 预计需要约 {memory['required_gb']:.1f}GB，"
                f"本机约 {memory['system_gb']:.1f}GB；请降低模型档位或更换设备。"
            )
    model = OrbitForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    total_steps = max(train_cfg.steps, 1)
    def lr_lambda(step):
        if step < train_cfg.warmup_steps:
            return max(step, 1) / max(train_cfg.warmup_steps, 1)
        if train_cfg.scheduler == "constant":
            return 1.0
        progress = (step - train_cfg.warmup_steps) / max(total_steps - train_cfg.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if resume:
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_step = int(state.get("step", 0))
    corpus = text or ("Orbit learns patterns from examples. Train small, test often, and share useful models. " * 200)
    model.train()
    for step in range(start_step + 1, train_cfg.steps + 1):
        if stop_event is not None and stop_event.is_set():
            break
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = byte_batch(corpus, train_cfg.batch_size, train_cfg.seq_len, device)
            with _autocast(device, train_cfg.precision):
                result = model(x, targets=y)
                loss = result["loss"] / train_cfg.grad_accum
            loss.backward()
            loss_value += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if callback:
            callback(step, loss_value)
        if train_cfg.checkpoint_every and step % train_cfg.checkpoint_every == 0:
            _save(checkpoint, model, optimizer, scheduler, cfg, preset, train_cfg, step)
    _save(checkpoint, model, optimizer, scheduler, cfg, preset, train_cfg, step if 'step' in locals() else start_step)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Orbit model locally")
    parser.add_argument("--preset", choices=["300m", "1b", "3b", "7b", "14b", "38b", "local"], default="local")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("orbit-checkpoint.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    data_path = args.data or args.text
    text = data_path.read_text(encoding="utf-8") if data_path else ""
    run_training(
        steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        learning_rate=args.lr, device_name=args.device, checkpoint=args.checkpoint,
        text=text, preset=args.preset, grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, precision=args.precision, scheduler_name=args.scheduler,
        checkpoint_every=args.checkpoint_every, seed=args.seed, resume=args.resume,
    )
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
