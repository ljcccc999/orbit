"""Deterministic synthetic worlds for testing object permanence and causality."""

from __future__ import annotations

import random
from dataclasses import dataclass


COLORS = ("red", "blue", "green", "yellow")
PLACES = ("left", "middle", "right")


@dataclass(frozen=True)
class WorldExample:
    observations: tuple[str, ...]
    action: str
    answer: str
    state_before: tuple[tuple[str, str], ...]
    state_after: tuple[tuple[str, str], ...]


def generate_world(seed: int) -> WorldExample:
    rng = random.Random(seed)
    colors = rng.sample(COLORS, 2)
    positions = {colors[0]: rng.choice(PLACES), colors[1]: rng.choice(PLACES)}
    hidden = rng.choice(colors)
    moving = rng.choice(colors)
    destination = rng.choice([p for p in PLACES if p != positions[moving]])
    observations = (
        f"the {colors[0]} object is at {positions[colors[0]]}",
        f"the {colors[1]} object is at {positions[colors[1]]}",
        f"the {hidden} object is hidden",
    )
    before = tuple(sorted(positions.items()))
    positions[moving] = destination
    after = tuple(sorted(positions.items()))
    return WorldExample(
        observations=observations,
        action=f"move {moving} to {destination}",
        answer=f"{moving} is at {destination}",
        state_before=before,
        state_after=after,
    )


def split_seeds(count: int, validation_fraction: float = 0.2):
    boundary = int(count * (1.0 - validation_fraction))
    return range(boundary), range(boundary, count)

