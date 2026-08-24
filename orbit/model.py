"""Orbit model: a compact hybrid-attention sparse language model."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import OrbitConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SiTUGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate_value = self.gate(x)
        gate = 4.0 * torch.tanh(gate_value / 4.0) * torch.sigmoid(gate_value)
        up_value = self.up(x)
        up = 25.0 * torch.tanh(up_value / 25.0)
        return self.down(gate * up)


class ShortConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)


class DeltaAttention(nn.Module):
    def __init__(self, cfg: OrbitConfig):
        super().__init__()
        d, h, k = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.n_heads, self.head_dim = h, k
        self.fast_attention = bool(getattr(cfg, "fast_attention", True))
        self.q_proj = nn.Linear(d, h * k, bias=False)
        self.k_proj = nn.Linear(d, h * k, bias=False)
        self.v_proj = nn.Linear(d, h * k, bias=False)
        self.q_conv, self.k_conv, self.v_conv = ShortConv(h * k), ShortConv(h * k), ShortConv(h * k)
        self.beta = nn.Linear(d, h)
        self.decay = nn.Linear(d, h * k)
        self.log_scale = nn.Parameter(torch.zeros(h))
        self.gate, self.out = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.out_norm = RMSNorm(k)

    def _heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = F.normalize(self._heads(F.silu(self.q_conv(self.q_proj(x)))), dim=-1)
        k = F.normalize(self._heads(F.silu(self.k_conv(self.k_proj(x)))), dim=-1)
        v = self._heads(F.silu(self.v_conv(self.v_proj(x))))
        if getattr(self, "fast_attention", False):
            # This keeps the same projections and causal ordering while using
            # PyTorch's device kernel instead of a Python loop over tokens.
            # It is materially faster on MPS and CUDA, especially with the
            # half-precision training path.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2)
            out = self.out_norm(out).reshape(b, t, -1)
            return self.out(torch.sigmoid(self.gate(x)) * out)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2).unsqueeze(-1)
        decay_logits = self.decay(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.log_scale.exp().view(1, self.n_heads, 1, 1)
        alpha = torch.exp(-5.0 * torch.sigmoid(scale * decay_logits))
        state = x.new_zeros(b, self.n_heads, self.head_dim, self.head_dim)
        outputs = []
        for i in range(t):
            qi, ki, vi = q[:, :, i], k[:, :, i], v[:, :, i]
            projected = torch.einsum("bhk,bhkv->bhv", ki * alpha[:, :, i], state)
            bi = beta[:, :, i].unsqueeze(-1)
            state = state - bi * ki.unsqueeze(-1) * projected.unsqueeze(-2)
            state = state + bi * ki.unsqueeze(-1) * vi.unsqueeze(-2)
            outputs.append(torch.einsum("bhkv,bhk->bhv", state, qi))
        out = torch.stack(outputs, dim=2).transpose(1, 2)
        out = self.out_norm(out).reshape(b, t, -1)
        return self.out(torch.sigmoid(self.gate(x)) * out)


class GatedMLA(nn.Module):
    def __init__(self, cfg: OrbitConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.latent = nn.Linear(cfg.d_model, cfg.latent_dim, bias=False)
        self.k_up = nn.Linear(cfg.latent_dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.v_up = nn.Linear(cfg.latent_dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.gate, self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False), nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        latent = self.latent(x)
        k = self.k_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.out(torch.sigmoid(self.gate(x)) * y)


class LatentMoE(nn.Module):
    def __init__(self, cfg: OrbitConfig):
        super().__init__()
        self.n_experts, self.top_k = cfg.n_routed_experts, cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)
        self.down, self.up = nn.Linear(cfg.d_model, cfg.latent_dim, bias=False), nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.latent_norm = RMSNorm(cfg.latent_dim)
        self.shared = nn.ModuleList([SiTUGLU(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_shared_experts)])
        self.routed = nn.ModuleList([SiTUGLU(cfg.latent_dim, cfg.expert_hidden) for _ in range(cfg.n_routed_experts)])
        self.last_balance_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.router(x)
        values, indices = logits.topk(self.top_k, dim=-1)
        weights = values.softmax(dim=-1)
        # MPS autocast can keep router logits in FP16 while top-k softmax is
        # accumulated in FP32.  Scatter requires matching dtypes.
        probs = torch.zeros_like(logits).scatter(-1, indices, weights.to(logits.dtype))
        z = self.down(x)
        # Only top-k experts have non-zero routing weights.  Evaluating every
        # expert for every token made the sparse MoE dense in practice.
        flat_z = z.reshape(-1, z.shape[-1])
        flat_probs = probs.reshape(-1, probs.shape[-1])
        routed_flat = torch.zeros_like(flat_z)
        for expert_id, expert in enumerate(self.routed):
            weights = flat_probs[:, expert_id]
            selected = torch.nonzero(weights > 0, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            values = expert(flat_z.index_select(0, selected))
            values = values * weights.index_select(0, selected).unsqueeze(-1)
            routed_flat = routed_flat.index_add(0, selected, values)
        routed = routed_flat.reshape(*x.shape[:-1], self.down.out_features)
        y = self.up(self.latent_norm(routed))
        for expert in self.shared:
            y = y + expert(x)
        load = probs.mean(dim=tuple(range(probs.ndim - 1)))
        self.last_balance_loss = ((load * self.n_experts) ** 2).mean()
        return y


class DepthResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.norm = RMSNorm(dim)

    def forward(self, current: Tensor, sources: list[Tensor]) -> Tensor:
        stacked = torch.stack(sources, dim=2)
        scores = torch.einsum("d,btsd->bts", self.query, self.norm(stacked))
        return current + torch.einsum("bts,btsd->btd", scores.softmax(-1), stacked)


class OrbitBackbone(nn.Module):
    def __init__(self, cfg: OrbitConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers, self.norms, self.depth = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(cfg.n_layers):
            attention = GatedMLA(cfg) if i % 4 == 3 or i == cfg.n_layers - 1 else DeltaAttention(cfg)
            self.layers.append(nn.ModuleDict({"attention": attention, "moe": LatentMoE(cfg)}))
            self.norms.append(nn.ModuleList([RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)]))
            self.depth.append(DepthResidual(cfg.d_model))
        self.final_norm = RMSNorm(cfg.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        h, sources = self.embedding(input_ids), []
        sources.append(h)
        block_sum: Optional[Tensor] = None
        losses = []
        for i, layer in enumerate(self.layers):
            if i % self.cfg.attnres_block_size == 0:
                block_sum = None
            depth_sources = sources if block_sum is None else sources + [block_sum]
            h = self.depth[i](h, depth_sources)
            h = h + layer["attention"](self.norms[i][0](h))
            h = h + layer["moe"](self.norms[i][1](h))
            losses.append(layer["moe"].last_balance_loss)
            block_sum = h if block_sum is None else block_sum + h
            if (i + 1) % self.cfg.attnres_block_size == 0 or i == len(self.layers) - 1:
                sources.append(block_sum)
        self.last_balance_loss = torch.stack(losses).mean()
        return self.final_norm(h)


class OrbitForCausalLM(nn.Module):
    def __init__(self, cfg: Optional[OrbitConfig] = None):
        super().__init__()
        self.cfg = cfg or OrbitConfig.tiny()
        self.backbone = OrbitBackbone(self.cfg)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids: Tensor, targets: Optional[Tensor] = None) -> dict[str, Tensor]:
        hidden = self.backbone(input_ids)
        logits = F.linear(hidden, self.lm_head.weight)
        result = {"logits": logits, "router_loss": self.backbone.last_balance_loss}
        if targets is not None:
            result["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            result["loss"] = result["loss"] + 0.01 * result["router_loss"]
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        vocab_limit: Optional[int] = None,
    ) -> Tensor:
        self.eval()
        limit = self.cfg.vocab_size if vocab_limit is None else min(int(vocab_limit), self.cfg.vocab_size)
        if limit < 1:
            raise ValueError("vocab_limit must be positive")
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.cfg.max_seq_len :]
            logits = self(context)["logits"][:, -1, :limit]
            if temperature <= 0:
                next_token = logits.argmax(-1, keepdim=True)
            else:
                next_token = torch.multinomial((logits / temperature).softmax(-1), 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
