"""Minimal MLX smoke trainer.

This first trainer learns next-token prediction while exposing state and
transition tensors for upcoming supervised world losses. It is deliberately
small enough to validate execution on a Mac before investing in tokenization
and a large corpus pipeline.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .config import OrbitConfig
from .model import OrbitModel


def loss_fn(model, tokens):
    result = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return mx.mean(nn.losses.cross_entropy(result.logits, targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("tiny", "small", "7b"), default="tiny")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    config = OrbitConfig.preset(args.preset)
    model = OrbitModel(config)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(1, args.steps + 1):
        tokens = mx.random.randint(0, config.vocab_size, (args.batch_size, args.sequence_length))
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if step == 1 or step % 10 == 0:
            print(f"step={step} loss={float(loss):.4f}")


if __name__ == "__main__":
    main()
