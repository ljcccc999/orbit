from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrbitConfig:
    vocab_size: int = 4096
    width: int = 256
    perception_layers: int = 4
    decoder_layers: int = 4
    transition_layers: int = 3
    heads: int = 8
    kv_heads: int = 4
    slots: int = 8
    slot_iterations: int = 3
    imagination_steps: int = 4
    max_sequence_length: int = 512
    state_width: int = 256
    ff_multiplier: float = 3.5
    dropout: float = 0.0
    parameter_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if self.state_width != self.width:
            raise ValueError("prototype requires state_width == width")
        if self.slots < 1 or self.imagination_steps < 1:
            raise ValueError("slots and imagination_steps must be positive")
        if self.parameter_dtype not in ("float32", "float16"):
            raise ValueError("parameter_dtype must be float32 or float16")

    @classmethod
    def preset(cls, name: str) -> "OrbitConfig":
        presets = {
            "tiny": cls(),
            "small": cls(
                vocab_size=16_384, width=768, perception_layers=8,
                decoder_layers=8, transition_layers=6, heads=12, kv_heads=4,
                slots=16, max_sequence_length=2048, state_width=768,
            ),
            # A Mac inference target. Training it from scratch is intentionally
            # not presented as feasible on a single 24 GB machine.
            "7b": cls(
                vocab_size=65_536, width=4096, perception_layers=10,
                decoder_layers=12, transition_layers=5, heads=32, kv_heads=8,
                slots=32, imagination_steps=6, max_sequence_length=8192,
                state_width=4096, ff_multiplier=3.5, parameter_dtype="float16",
            ),
        }
        try:
            return presets[name]
        except KeyError as exc:
            raise ValueError(f"unknown preset {name!r}: {', '.join(presets)}") from exc

    def rough_parameter_count(self) -> int:
        """Conservative architecture estimate, independent of an MLX install."""
        w = self.width
        ff = int(w * self.ff_multiplier)
        # Attention projections + SwiGLU MLP per transformer block.
        per_block = 4 * w * w + 3 * w * ff
        transformer = (self.perception_layers + self.decoder_layers) * per_block
        # Transition blocks operate over slots with the same core shape.
        transition = self.transition_layers * per_block
        embeddings = 2 * self.vocab_size * w
        binding_and_state = 12 * w * w + self.slots * w
        return transformer + transition + embeddings + binding_and_state
