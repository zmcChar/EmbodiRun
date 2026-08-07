"""Swappable eager and fused scaled-dot-product attention operators.

Adapted from ``vvla/layers/attention.py`` at vvla commit
``80b5cf48c8710c69ed97200903562e9787efe105`` (MIT).  The prototype keeps only
the two operators needed to demonstrate the engine/backend boundary.
"""

from __future__ import annotations

import difflib
from typing import Any, Protocol

import torch
from torch.nn import functional


class AttentionOperator(Protocol):
    name: str

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor: ...


def _prepare_mask(
    attention_mask: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape[-1] != key.shape[-2]:
        attention_mask = attention_mask[..., : key.shape[-2]]
    if attention_mask.is_floating_point() and attention_mask.dtype != query.dtype:
        attention_mask = attention_mask.to(query.dtype)
    return attention_mask


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    batch, kv_heads, sequence, width = value.shape
    return (
        value[:, :, None, :, :]
        .expand(batch, kv_heads, repeats, sequence, width)
        .reshape(batch, kv_heads * repeats, sequence, width)
    )


def _validate_heads(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> int:
    if key.shape[1] != value.shape[1]:
        raise ValueError("key and value must have the same number of heads")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by key/value head count for GQA")
    return query.shape[1] // key.shape[1]


class EagerAttention:
    """Numerical reference: explicit matmul and fp32 softmax."""

    name = "eager"

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        repeats = _validate_heads(query, key, value)
        key = _repeat_kv(key, repeats)
        value = _repeat_kv(value, repeats)
        attention_mask = _prepare_mask(attention_mask, query, key)
        scale = query.shape[-1] ** -0.5 if scaling is None else scaling
        weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout_p:
            weights = functional.dropout(weights, p=dropout_p)
        return torch.matmul(weights, value)


class SDPAAttention:
    """Fused PyTorch SDPA, including grouped-query attention."""

    name = "sdpa"

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        repeats = _validate_heads(query, key, value)
        attention_mask = _prepare_mask(attention_mask, query, key)
        kwargs: dict[str, Any] = {
            "attn_mask": attention_mask,
            "scale": scaling,
            "dropout_p": dropout_p,
        }
        if repeats != 1:
            kwargs["enable_gqa"] = True
        return functional.scaled_dot_product_attention(
            query,
            key,
            value,
            **kwargs,
        )


_OPERATORS: dict[str, type[AttentionOperator]] = {
    EagerAttention.name: EagerAttention,
    SDPAAttention.name: SDPAAttention,
}


def get_attention_operator(name: str) -> AttentionOperator:
    try:
        operator = _OPERATORS[name]
    except KeyError as error:
        suggestion = difflib.get_close_matches(name, _OPERATORS, n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(
            f"unknown attention operator {name!r}{hint}; "
            f"available: {available_attention_operators()}"
        ) from error
    return operator()


def available_attention_operators() -> tuple[str, ...]:
    return tuple(sorted(_OPERATORS))
