"""Load a quantized Orbit 7B checkpoint and run one Metal forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbit.config import OrbitConfig
from orbit.model import OrbitModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", default="checkpoints/orbit-7b-random-4bit", nargs="?")
    parser.add_argument("--sequence-length", type=int, default=8)
    args = parser.parse_args()
    config = OrbitConfig.preset("7b")
    model = OrbitModel(config)
    nn.quantize(model, bits=4, group_size=64)
    model.load_weights(list(mx.load(str(Path(args.checkpoint) / "weights.safetensors")).items()))
    tokens = mx.random.randint(0, config.vocab_size, (1, args.sequence_length))
    output = model(tokens)
    mx.eval(output)
    print("Orbit 7B OCA Metal forward: OK")
    print("logits:", output.logits.shape)
    print("continuum state:", output.state.shape)
    print("imagined states:", output.imagined_states.shape)


if __name__ == "__main__":
    main()

