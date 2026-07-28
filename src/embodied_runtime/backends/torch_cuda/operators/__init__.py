"""Swappable PyTorch operator implementations."""

from .attention import (
    EagerAttention,
    SDPAAttention,
    available_attention_operators,
    get_attention_operator,
)

__all__ = [
    "EagerAttention",
    "SDPAAttention",
    "available_attention_operators",
    "get_attention_operator",
]
