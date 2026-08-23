# OCA-Research — Orbit Continuum Architecture

## Status and product boundary

OCA is a **self-developed, research-stage world-model architecture**. It is
not implemented as a finished world model and it has not been shown to
understand the real physical world. The current prototype and synthetic-world
tests only verify limited code paths and research hypotheses; they are not
evidence of general physical-world understanding.

Orbit's research/product goal is for OCA to become an industry-first
architecture for understanding the physical world. “Industry-first” is a
goal/claim that still requires prior-art review, independent benchmarks and
reproducible third-party validation; it must not be presented as an established
fact today.

This directory belongs to the independent desktop Orbit project at
`/Users/tim/Desktop/YUNSH/Orbit/orbit/OCA-Research/`. OCA is separate from the
desktop Orbit chat model (`orbit-hybrid-moe-v1`), its checkpoints, training
history and runtime. The desktop app does not load OCA as its ordinary chat
model.

Orbit is an experimental AI system built around the **Orbit Continuum
Architecture (OCA)**. OCA treats language as an observation of a changing
world, not as the world itself. It keeps a persistent latent state, binds
observations into object-like slots, learns state transitions, and can imagine
several future states before producing an answer.

This repository contains the architecture prototype, a small synthetic world
dataset, a training entry point for Apple silicon, and a scalable 7B
configuration. The 7B configuration is a specification: start with `tiny` on a
Mac and only scale after the architecture passes the included world-state
tests.

## OCA in one pass

```text
observation tokens -> perception blocks -> object-slot binding
                                           |
previous continuum state -> gated state update -> present world state
                                                   |
                                   causal transition model
                                      |      |      |
                                     t+1    t+2    t+n
                                      \      |      /
                                  imagined futures
                                           |
                              answer/action token decoder
```

OCA separates four functions that a conventional decoder-only model mixes
together:

1. **Perception** turns an observation into features.
2. **Binding** groups features into persistent object-like slots.
3. **Continuum state** carries information across episodes and time.
4. **Causal imagination** predicts how actions change that state.

The state is explicit and inspectable. Callers decide when to retain, detach,
or reset it; it is never silently carried between unrelated users.

## Quick start on Apple silicon

Use Python 3.10+ and MLX. No NVIDIA GPU is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m orbit.train --preset tiny --steps 200
python -m unittest discover -s tests

# Create the 7B architecture manifest without allocating its weights
python scripts/build_7b.py

# Run the locally generated 4-bit 7B structure on Metal
python scripts/smoke_7b.py
```

To allocate a float16 random 7B checkpoint on the Mac, use
`python scripts/build_7b.py --initialize`. It is an untrained structural
checkpoint and will be roughly 14 GiB before optimizer state; it is not a
finished assistant until trained.

The first benchmark scaffold is deliberately small: colored objects move
between locations, can be hidden, and must still be tracked. The current smoke
trainer verifies end-to-end language loss and Metal backpropagation. The next
research milestone is wiring the benchmark to explicit current-state and
future-state losses, so success is not measured only by fluent text.

See [docs/architecture.md](docs/architecture.md) for the design and research
criteria.

Formal paper versions:

- [English paper in Transformer-style structure](docs/OCA-Architecture-Paper-English-Transformer-Format.md)
- [中文论文（Transformer 风格结构）](docs/OCA-架构论文-中文版-Transformer格式.md)
