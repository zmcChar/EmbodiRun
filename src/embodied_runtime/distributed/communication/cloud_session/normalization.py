"""Strict JSON normalization for cloud-session request values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from embodied_runtime.models.request import RawRequest


def json_compatible(value: Any, *, path: str = "value") -> Any:
    """Convert tensor-like trees to strict JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return json_compatible(value.value, path=path)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            converted[key] = json_compatible(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [json_compatible(item, path=f"{path}[{index}]") for index, item in enumerate(value)]

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        return json_compatible(tolist(), path=path)
    item = getattr(candidate, "item", None)
    if callable(item):
        return json_compatible(item(), path=path)
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def prepare_request_values(
    raw: RawRequest,
    request_metadata: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    return (
        json_compatible(raw.observation, path="observation"),
        json_compatible(request_metadata, path="request_metadata"),
        json_compatible(raw.metadata, path="raw_metadata"),
    )
