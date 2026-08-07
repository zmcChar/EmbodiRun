"""Shared value normalization for evaluation data models."""

from __future__ import annotations

import math
from collections.abc import Sequence


def non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > 256:
        raise ValueError(f"{name} must be at most 256 characters")
    return normalized


def finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def positive_integer(name: str, value: int) -> None:
    non_negative_integer(name, value)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")


def latencies(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    normalized = tuple(finite_number(name, value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{name} values must be non-negative")
    return normalized
