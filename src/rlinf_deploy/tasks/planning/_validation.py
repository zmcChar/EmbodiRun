"""Validation helpers shared by task-planning value objects."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

from rlinf_deploy.types import Metadata, TensorTree

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def identifier(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{name} must be 1-128 characters using letters, digits, '.', '_', ':', or '-'"
        )
    return normalized


def text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def optional_text(name: str, value: str | None) -> str | None:
    return None if value is None else text(name, value)


def timestamp(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def metadata(value: Metadata) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return dict(value)


def observation_snapshot(value: TensorTree) -> TensorTree:
    try:
        return copy.deepcopy(value)
    except Exception as error:
        raise TypeError("observation must support a safe snapshot") from error
