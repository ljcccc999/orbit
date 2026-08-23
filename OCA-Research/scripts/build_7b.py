"""Build and optionally initialize the Orbit 7B architecture.

By default this creates a reproducible model manifest without allocating 7B
weights. `--initialize` allocates float16 weights, which is the appropriate
Mac-side representation for a 24 GB machine. A random checkpoint is a
structural artifact, not a pretrained conversational model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

# Allow `python scripts/build_7b.py` directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbit.checkpoint import save_checkpoint
from orbit.config import OrbitConfig
from orbit.model import OrbitModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/orbit-7b-init")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--quantize", action="store_true", help="4-bit Linear quantization after initialization")
    args = parser.parse_args()

    config = OrbitConfig.preset("7b")
    estimate = config.rough_parameter_count()
    manifest = {
        "model": "Orbit",
        "architecture": "Orbit Continuum Architecture",
        "preset": "7b",
        "estimated_parameters": estimate,
        "estimated_fp16_weight_gib": round(estimate * 2 / 1024**3, 2),
        "status": "architecture-ready; untrained" if not args.initialize else "initialized; untrained",
        "training_warning": "Random initialization is not a useful language model.",
        "config": config.__dict__,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.initialize:
        (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    model = OrbitModel(config)
    model.set_dtype(mx.float16)
    if args.quantize:
        import mlx.nn as nn
        nn.quantize(model, bits=4, group_size=64)
    mx.eval(model.parameters())
    info = save_checkpoint(
        model,
        config,
        output,
        metadata={
            "status": "initialized; untrained",
            "estimated_logical_parameters": estimate,
            "quantized_bits": 4 if args.quantize else None,
        },
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
