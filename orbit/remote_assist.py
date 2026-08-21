from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from threading import Event

from .teacher import TeacherConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an AI-assisted Orbit dataset on a remote GPU host")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    api_key = os.environ.get("ORBIT_TEACHER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set ORBIT_TEACHER_API_KEY before running an AI-assisted remote job")
    teacher = TeacherConfig(
        base_url=str(config.get("base_url", "")),
        model=str(config.get("model", "")),
        instruction=str(config.get("instruction", "")),
        examples=int(config.get("examples", 20)),
        language=str(config.get("language", "bilingual")),
        model_profile=config.get("model_profile") or {},
    )

    def progress(current: int, total: int) -> None:
        print(f"teacher samples: {current}/{total}", flush=True)

    text, usage = generate_dataset(teacher, api_key, Event(), progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"teacher usage: {json.dumps(usage, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
