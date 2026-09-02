"""Robot-to-model runtime binding configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation import string


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    runtime_id: str
    robot: str
    model: str
    binding: str
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_runtime(runtime_id: str, value: dict[str, Any]) -> RuntimeConfig:
    context = f"runtimes.{runtime_id}"
    return RuntimeConfig(
        runtime_id=runtime_id,
        robot=string(value, "robot", context),
        model=string(value, "model", context),
        binding=string(value, "binding", context),
        options=dict(value),
    )


__all__ = ["RuntimeConfig"]
