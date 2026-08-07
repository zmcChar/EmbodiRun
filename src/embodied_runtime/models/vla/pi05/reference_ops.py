"""Torch reference operators shared by the staged pi0.5 module."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    return residual + update if gate is None else residual + update * gate


def rmsnorm(norm: Any, x: torch.Tensor, condition: torch.Tensor | None):
    """PiGemma RMSNorm semantics, including optional adaRMS modulation."""

    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + norm.eps)
    if condition is None or norm.dense is None:
        return (normalized * (1.0 + norm.weight.float())).type_as(x), None
    modulation = norm.dense(condition)
    if x.ndim == 3:
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    normalized = normalized * (1 + scale.float()) + shift.float()
    return normalized.to(x.dtype), gate.to(x.dtype)


def mlp(mlp_module: Any, x: torch.Tensor) -> torch.Tensor:
    return mlp_module.down_proj(mlp_module.act_fn(mlp_module.gate_proj(x)) * mlp_module.up_proj(x))


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Plain Torch attention matching Hugging Face Gemma eager semantics."""

    repetitions = query.shape[1] // key.shape[1]
    key = _repeat_kv(key, repetitions)
    value = _repeat_kv(value, repetitions)
    if mask is not None:
        mask = mask[..., : key.shape[-2]]
        if mask.is_floating_point() and mask.dtype != query.dtype:
            mask = mask.to(query.dtype)
    attention = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if mask is not None:
        attention = attention + mask
    attention = F.softmax(attention, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attention, value)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(x: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return x
    batch, kv_heads, sequence, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, kv_heads, repetitions, sequence, head_dim)
        .reshape(batch, kv_heads * repetitions, sequence, head_dim)
    )
