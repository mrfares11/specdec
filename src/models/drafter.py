"""EAGLE/HASS-style drafter: a thin feature-autoregressive head on the target.

The drafter autoregresses over (token, previous-position feature) pairs:

    input_i  = FC([ embed(t_i) ; f_{i-1} ])   →   decoder layer(s)   →   f̂_i
    p(t_{i+1}) = lm_head(final_norm(f̂_i))

where ``f`` are target features (``hidden_states[data.layer_index]``). The
embedding, final norm, and lm-head are the FROZEN target's modules, bound via
:meth:`Drafter.bind_target`; they are deliberately kept out of the drafter's
``state_dict()``/``parameters()`` so the optimizer only ever sees drafter
weights (FC + its own decoder layers). The target is never trained.

The decoder layer is implemented self-contained (RMSNorm, RoPE, GQA attention
via SDPA, SwiGLU MLP) rather than reusing HF internals, so the module is stable
across transformers versions and fully CPU-testable.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn


class RMSNorm(nn.Module):
    """Llama-style RMS norm (computed in fp32, cast back)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _rope_cos_sin(
    position_ids: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [B, T, head_dim] for the given absolute positions."""
    device = position_ids.device
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    ang = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq  # [B, T, D/2]
    ang = torch.cat([ang, ang], dim=-1)  # [B, T, D]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary embedding on [B, heads, T, head_dim] tensors; cos/sin are [B, T, head_dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@dataclass
class DrafterCache:
    """Per-layer (k, v) tensors of shape [B, kv_heads, S, head_dim] (post-RoPE keys)."""

    layers: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def seq_length(self) -> int:
        """Number of cached positions (0 when empty)."""
        if not self.layers:
            return 0
        return int(self.layers[0][0].shape[2])

    def crop(self, length: int) -> None:
        """Drop cached positions beyond ``length`` (e.g. rejected draft nodes)."""
        self.layers = [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in self.layers]


class _Attention(nn.Module):
    """GQA attention with RoPE and an optional additive mask, via SDPA."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        s_past = 0
        if past is not None:
            s_past = int(past[0].shape[2])
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)

        rep = self.num_heads // self.num_kv_heads
        k_full = k.repeat_interleave(rep, dim=1) if rep > 1 else k
        v_full = v.repeat_interleave(rep, dim=1) if rep > 1 else v

        # No explicit bias and no cache → plain causal self-attention (training).
        is_causal = attn_bias is None and past is None and t > 1
        if attn_bias is None and s_past > 0 and t > 1:
            # Cached multi-token extension: past fully visible, causal within
            # the new block. Without this, SDPA would run UNMASKED here and
            # extension tokens would attend later tokens in the same block
            # (latent for 1-layer drafters, corrupting for >= 2 layers).
            neg = torch.finfo(x.dtype).min
            attn_bias = torch.zeros(1, 1, t, s_past + t, dtype=x.dtype, device=x.device)
            block = torch.full((t, t), neg, dtype=x.dtype, device=x.device).triu(1)
            attn_bias[0, 0, :, s_past:] = block
        out = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=attn_bias, is_causal=is_causal
        )
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out), (k, v)


class _MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderLayer(nn.Module):
    """Pre-norm transformer block.

    ``no_input_norm=True`` replaces the attention input norm with identity —
    the official EAGLE drafters build their FIRST layer this way (the FC output
    feeds attention un-normalized), so importing their weights requires it.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        no_input_norm: bool = False,
    ):
        super().__init__()
        self.input_norm = nn.Identity() if no_input_norm else RMSNorm(hidden_size)
        self.attn = _Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attn_norm = RMSNorm(hidden_size)
        self.mlp = _MLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, kv = self.attn(self.input_norm(x), cos, sin, attn_bias, past)
        x = x + h
        x = x + self.mlp(self.post_attn_norm(x))
        return x, kv


class Drafter(nn.Module):
    """The trainable drafter. See module docstring for the computation.

    Construct via :meth:`from_config`, then :meth:`bind_target` before use.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int = 1,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rope_theta: float = 500000.0,
        apply_final_norm: bool = True,
        fc_bias: bool = False,
        layer0_no_input_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rope_theta = rope_theta
        self.apply_final_norm = apply_final_norm
        self.head_dim = hidden_size // num_heads
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=fc_bias)
        self.layers = nn.ModuleList(
            _DecoderLayer(
                hidden_size, num_heads, num_kv_heads, intermediate_size,
                no_input_norm=(i == 0 and layer0_no_input_norm),
            )
            for i in range(num_layers)
        )
        # Frozen target modules, bound later. Held in a plain list so nn.Module
        # does NOT register them: they stay out of state_dict()/parameters().
        self._bound: list = []

    @classmethod
    def from_config(cls, drafter_cfg: DictConfig) -> "Drafter":
        """Build from ``cfg.drafter``."""
        return cls(
            hidden_size=int(drafter_cfg.hidden_size),
            num_layers=int(drafter_cfg.num_layers),
            num_heads=int(drafter_cfg.num_heads),
            num_kv_heads=int(drafter_cfg.num_kv_heads),
            intermediate_size=int(drafter_cfg.intermediate_size),
            rope_theta=float(drafter_cfg.rope_theta),
            apply_final_norm=bool(drafter_cfg.apply_final_norm),
            fc_bias=bool(drafter_cfg.get("fc_bias", False)),
            layer0_no_input_norm=bool(drafter_cfg.get("layer0_no_input_norm", False)),
        )

    def bind_target(self, target: nn.Module, cast_dtype: Optional[torch.dtype] = None) -> None:
        """Borrow the frozen target's embedding, final norm, and lm-head.

        These stay frozen and unregistered — ``parameters()`` still yields only
        drafter weights, so optimizers cannot touch the target.

        Args:
            target: the frozen verifier.
            cast_dtype: when the target is loaded in reduced precision (pod
                training loads it bf16 to halve VRAM), pass ``torch.float32``
                to bind fp32 COPIES of the three modules: training math stays
                fp32 while the frozen bulk of the target keeps its compact
                dtype. Llama-3 checkpoints store these weights in bf16, so
                the upcast loses nothing vs loading the target in fp32.
        """
        embed = target.get_input_embeddings()
        head = target.get_output_embeddings()
        norm = target.model.norm
        if cast_dtype is not None:
            embed, norm, head = (
                copy.deepcopy(m).to(cast_dtype) for m in (embed, norm, head)
            )
        if embed.weight.shape[1] != self.hidden_size:
            raise ValueError(
                f"drafter hidden_size={self.hidden_size} != target hidden {embed.weight.shape[1]}"
            )
        for m in (embed, norm, head):
            m.requires_grad_(False)
        self._bound = [embed, norm, head]

    @property
    def _embed(self) -> nn.Module:
        if not self._bound:
            raise RuntimeError("call bind_target(target) before using the drafter")
        return self._bound[0]

    @property
    def _final_norm(self) -> nn.Module:
        return self._bound[1]

    @property
    def _lm_head(self) -> nn.Module:
        return self._bound[2]

    def forward(
        self,
        token_ids: torch.Tensor,
        prev_feats: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        cache: Optional[DrafterCache] = None,
    ) -> Tuple[torch.Tensor, DrafterCache]:
        """One drafter pass.

        Args:
            token_ids: long [B, T] — tokens ``t_i``.
            prev_feats: float [B, T, H] — features of the PREVIOUS position
                (``f_{i-1}``): true target features for committed context,
                drafter outputs for drafted nodes.
            position_ids: long [B, T] — absolute positions (context_len + depth
                for tree nodes).
            attn_bias: additive float mask [B, 1, T, S_total]; None → causal.
            cache: existing :class:`DrafterCache` to append to (its contents are
                not modified; an extended cache is returned).

        Returns:
            (features [B, T, H], extended cache).
        """
        emb = self._embed(token_ids)
        x = self.fc(torch.cat([emb.to(prev_feats.dtype), prev_feats], dim=-1))
        cos, sin = _rope_cos_sin(position_ids, self.head_dim, self.rope_theta, x.dtype)

        new_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past = cache.layers[i] if cache is not None and cache.layers else None
            x, kv = layer(x, cos, sin, attn_bias, past)
            new_layers.append(kv)
        return x, DrafterCache(new_layers)

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Next-token logits from drafter features via the frozen target head."""
        h = self._final_norm(feats) if self.apply_final_norm else feats
        return self._lm_head(h)

    def num_trainable_parameters(self) -> int:
        """Trainable parameter count (bound target modules excluded by design)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
