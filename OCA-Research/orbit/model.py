"""MLX implementation of Orbit Continuum Architecture.

The architecture is intentionally explicit: perception, binding, persistent
state, causal transition, and decoding are separate modules so each claim can
be ablated and measured.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .config import OrbitConfig


class OrbitOutput(NamedTuple):
    logits: mx.array
    state: mx.array
    slots: mx.array
    imagined_states: mx.array


def _dtype(name: str):
    return mx.float16 if name == "float16" else mx.float32


class SwiGLU(nn.Module):
    def __init__(self, width: int, multiplier: float, parameter_dtype: str = "float32"):
        super().__init__()
        hidden = int(width * multiplier)
        hidden = ((hidden + 63) // 64) * 64
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)
        self.set_dtype(_dtype(parameter_dtype))

    def __call__(self, x):
        return self.out(nn.silu(self.gate(x)) * self.value(x))


class AttentionBlock(nn.Module):
    def __init__(self, config: OrbitConfig, causal: bool):
        super().__init__()
        self.width = config.width
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.causal = causal
        self.norm1 = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)
        self.norm2 = nn.RMSNorm(config.width)
        self.ff = SwiGLU(config.width, config.ff_multiplier, config.parameter_dtype)
        self.set_dtype(_dtype(config.parameter_dtype))

    def __call__(self, x):
        residual = x
        x = self.norm1(x)
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        if self.causal:
            mask = mx.triu(mx.full((length, length), -1e9), k=1)
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, self.width)
        x = residual + self.proj(attended)
        return x + self.ff(self.norm2(x))


class SlotBinder(nn.Module):
    """Iterative competitive binding into object-like latent slots."""

    def __init__(self, config: OrbitConfig):
        super().__init__()
        self.iterations = config.slot_iterations
        self.scale = config.width ** -0.5
        self.slot_seed = mx.zeros((config.slots, config.width))
        self.to_q = nn.Linear(config.width, config.width, bias=False)
        self.to_k = nn.Linear(config.width, config.width, bias=False)
        self.to_v = nn.Linear(config.width, config.width, bias=False)
        self.candidate = nn.Linear(2 * config.width, config.width)
        self.gate = nn.Linear(2 * config.width, config.width)
        self.norm_features = nn.RMSNorm(config.width)
        self.norm_slots = nn.RMSNorm(config.width)
        self.set_dtype(_dtype(config.parameter_dtype))

    def __call__(self, features):
        batch = features.shape[0]
        slots = mx.broadcast_to(self.slot_seed[None], (batch,) + self.slot_seed.shape)
        keys = self.to_k(self.norm_features(features))
        values = self.to_v(features)
        for _ in range(self.iterations):
            queries = self.to_q(self.norm_slots(slots))
            affinity = mx.einsum("bsd,btd->bst", queries, keys) * self.scale
            # Competition over slots makes features choose an object hypothesis.
            assignment = mx.softmax(affinity, axis=1) + 1e-6
            assignment = assignment / mx.sum(assignment, axis=-1, keepdims=True)
            updates = mx.einsum("bst,btd->bsd", assignment, values)
            joined = mx.concatenate([slots, updates], axis=-1)
            gate = mx.sigmoid(self.gate(joined))
            proposal = mx.tanh(self.candidate(joined))
            slots = gate * slots + (1.0 - gate) * proposal
        return slots


class ContinuumUpdater(nn.Module):
    """Updates persistent state without conflating it with token KV cache."""

    def __init__(self, width: int, parameter_dtype: str = "float32"):
        super().__init__()
        self.update_gate = nn.Linear(2 * width, width)
        self.reset = nn.Linear(2 * width, width)
        self.candidate = nn.Linear(2 * width, width)
        self.set_dtype(_dtype(parameter_dtype))

    def __call__(self, previous, observed):
        joined = mx.concatenate([previous, observed], axis=-1)
        update = mx.sigmoid(self.update_gate(joined))
        reset = mx.sigmoid(self.reset(joined))
        candidate_input = mx.concatenate([reset * previous, observed], axis=-1)
        candidate = mx.tanh(self.candidate(candidate_input))
        return (1.0 - update) * previous + update * candidate


class CausalTransition(nn.Module):
    """Action-conditioned state dynamics used for counterfactual rollout."""

    def __init__(self, config: OrbitConfig):
        super().__init__()
        self.action = nn.Linear(config.width, config.width, bias=False)
        self.blocks = [AttentionBlock(config, causal=False) for _ in range(config.transition_layers)]
        self.delta = nn.Linear(config.width, config.width)
        self.confidence = nn.Linear(config.width, 1)
        self.set_dtype(_dtype(config.parameter_dtype))

    def __call__(self, state, action):
        x = state + self.action(action)[:, None, :]
        for block in self.blocks:
            x = block(x)
        confidence = mx.sigmoid(self.confidence(x))
        return state + confidence * mx.tanh(self.delta(x))


class OrbitModel(nn.Module):
    def __init__(self, config: OrbitConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.width)
        self.perception = [AttentionBlock(config, causal=False) for _ in range(config.perception_layers)]
        self.binder = SlotBinder(config)
        self.state_update = ContinuumUpdater(config.width, config.parameter_dtype)
        self.transition = CausalTransition(config)
        self.decoder = [AttentionBlock(config, causal=True) for _ in range(config.decoder_layers)]
        self.state_to_decoder = nn.Linear(config.width, config.width, bias=False)
        self.final_norm = nn.RMSNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.set_dtype(_dtype(config.parameter_dtype))

    def initial_state(self, batch_size: int):
        return mx.zeros((batch_size, self.config.slots, self.config.width))

    def __call__(self, tokens, previous_state=None, action=None):
        batch, length = tokens.shape
        if length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds configured maximum")
        positions = mx.arange(length)
        features = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        for block in self.perception:
            features = block(features)

        observed_slots = self.binder(features)
        if previous_state is None:
            previous_state = self.initial_state(batch)
        state = self.state_update(previous_state, observed_slots)

        if action is None:
            action = mx.mean(features, axis=1)
        imagined = []
        future = state
        for _ in range(self.config.imagination_steps):
            future = self.transition(future, action)
            imagined.append(future)
        imagined_states = mx.stack(imagined, axis=1)

        context = self.state_to_decoder(mx.mean(state, axis=1))[:, None, :]
        decoded = features + context
        for block in self.decoder:
            decoded = block(decoded)
        logits = self.output(self.final_norm(decoded))
        return OrbitOutput(logits, state, observed_slots, imagined_states)
