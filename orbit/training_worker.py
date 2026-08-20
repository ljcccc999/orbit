from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    stop_event = threading.Event()
    try:
        job = json.loads(args.job.read_text(encoding="utf-8"))
        from .train import run_training, select_device
        from .training_config import TrainingConfig

        device = select_device(str(job["device"]))
        send({"type": "ready", "device": device.type})

        def commands() -> None:
            for line in sys.stdin:
                try:
                    if json.loads(line).get("command") == "stop":
                        stop_event.set()
                        return
                except json.JSONDecodeError:
                    continue

        threading.Thread(target=commands, name="orbit-training-commands", daemon=True).start()

        def progress(step: int, loss: float) -> None:
            send({"type": "progress", "step": step, "loss": loss})

        run_training(
            device_name=str(job["device"]), checkpoint=Path(job["checkpoint"]),
            text=Path(job["dataset"]).read_text(encoding="utf-8"), preset=str(job["preset"]),
            callback=progress, stop_event=stop_event,
            training_config=TrainingConfig(**job["training_config"]),
            resume=Path(job["resume"]) if job.get("resume") else None,
            resume_weights_only=bool(job.get("resume")), model_metadata=job.get("metadata") or {},
        )
        send({"type": "stopped" if stop_event.is_set() else "completed"})
    except Exception as exc:
        send({"type": "fatal", "error": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
