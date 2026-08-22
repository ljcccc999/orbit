from __future__ import annotations

from dataclasses import dataclass, replace
import os
import subprocess


# The current local trainer is deliberately tokenizer-free: it trains on
# UTF-8 bytes.  Keep the model vocabulary aligned with byte_batch() instead
# of allocating a 32K output head whose classes can never be targets.
BYTE_VOCAB_SIZE = 256


@dataclass
class OrbitConfig:
    vocab_size: int = BYTE_VOCAB_SIZE
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 32
    latent_dim: int = 128
    expert_hidden: int = 256
    n_routed_experts: int = 8
    top_k: int = 2
    n_shared_experts: int = 1
    attnres_block_size: int = 4
    max_seq_len: int = 2048
    vision_width: int = 128
    vision_layers: int = 2
    vision_heads: int = 4
    vision_patch_size: int = 14
    # Use the fused causal attention implementation by default.  The previous
    # recurrent reference path is retained for compatibility/debugging, but it
    # performs a Python loop for every token and is not suitable for local
    # training on MPS.
    fast_attention: bool = True

    @classmethod
    def tiny(cls) -> "OrbitConfig":
        return cls(
            d_model=128, n_layers=4, n_heads=4, head_dim=32,
            latent_dim=64, expert_hidden=128, n_routed_experts=4,
            top_k=2, attnres_block_size=2, max_seq_len=512,
            vision_width=64, vision_layers=2,
        )

    @classmethod
    def one_billion(cls) -> "OrbitConfig":
        """Approximate 1B architecture description without allocating it."""
        return cls(
            vocab_size=BYTE_VOCAB_SIZE, d_model=1536, n_layers=28,
            n_heads=16, head_dim=96, latent_dim=512,
            expert_hidden=1280, n_routed_experts=8, top_k=2,
            n_shared_experts=1, attnres_block_size=4,
            max_seq_len=4096, vision_width=512, vision_layers=8,
            vision_heads=8,
        )

    @classmethod
    def presets(cls) -> dict[str, "OrbitConfig"]:
        return {
            "300m": cls(vocab_size=BYTE_VOCAB_SIZE, d_model=768, n_layers=28, n_heads=12, head_dim=64, latent_dim=256, expert_hidden=768, n_routed_experts=8, max_seq_len=2048),
            "1b": cls.one_billion(),
            "3b": cls(vocab_size=BYTE_VOCAB_SIZE, d_model=2048, n_layers=32, n_heads=16, head_dim=128, latent_dim=1024, expert_hidden=2048, n_routed_experts=8, max_seq_len=4096),
            "7b": cls(vocab_size=BYTE_VOCAB_SIZE, d_model=3072, n_layers=36, n_heads=24, head_dim=128, latent_dim=1024, expert_hidden=4096, n_routed_experts=8, max_seq_len=4096),
            "14b": cls(vocab_size=BYTE_VOCAB_SIZE, d_model=4096, n_layers=42, n_heads=32, head_dim=128, latent_dim=1536, expert_hidden=4608, n_routed_experts=8, max_seq_len=8192),
            "38b": cls(vocab_size=BYTE_VOCAB_SIZE, d_model=6144, n_layers=48, n_heads=48, head_dim=128, latent_dim=3072, expert_hidden=5888, n_routed_experts=8, max_seq_len=8192),
        }

    @classmethod
    def for_preset(cls, name: str) -> "OrbitConfig":
        if name == "tiny" or name == "local":
            return cls.tiny()
        try:
            return cls.presets()[name]
        except KeyError as exc:
            raise ValueError(f"unknown model preset: {name}") from exc

    def with_overrides(self, **kwargs) -> "OrbitConfig":
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if not 1 <= self.top_k <= self.n_routed_experts:
            raise ValueError("top_k must be in [1, n_routed_experts]")
        if self.n_layers < 1 or self.attnres_block_size < 1:
            raise ValueError("n_layers and attnres_block_size must be positive")

    def estimate_parameters(self) -> int:
        """Estimate parameter count without allocating the model."""
        d, h, k, l = self.d_model, self.n_heads, self.head_dim, self.latent_dim
        kda = 3 * d * d + d * h + d * (h * k) + 2 * d * d
        mla = d * d + d * l + 2 * l * h * k + 2 * d * d
        attention = (3 * kda + mla) / 4
        shared = self.n_shared_experts * 3 * d * self.expert_hidden
        routed = 2 * d * l + self.n_routed_experts * 3 * l * self.expert_hidden
        moe = shared + routed + d * self.n_routed_experts
        return int(self.vocab_size * d + self.n_layers * (attention + moe))

    def estimated_training_memory_gb(self) -> float:
        # Includes optimizer state plus a conservative activation/workspace
        # allowance. The exact number depends on precision and checkpointing.
        return self.estimate_parameters() * 32 / 1_000_000_000

    @staticmethod
    def system_memory_gb() -> float:
        try:
            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(raw) / 1_000_000_000
        except (OSError, subprocess.SubprocessError, ValueError):
            try:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1_000_000_000
            except (ValueError, OSError):
                return 0.0

    def memory_check(self, safety_ratio: float = 0.75) -> dict[str, float | bool]:
        required = self.estimated_training_memory_gb()
        available = self.system_memory_gb()
        return {
            "required_gb": required,
            "system_gb": available,
            "can_train": bool(available <= 0 or required <= available * safety_ratio),
        }
