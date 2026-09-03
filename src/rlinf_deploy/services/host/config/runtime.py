"""Robot-to-model runtime binding configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation import ConfigError, mapping, string


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    runtime_id: str
    robot: str
    model: str
    binding: str
    inputs: dict[str, str]
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_runtime(runtime_id: str, value: dict[str, Any]) -> RuntimeConfig:
    context = f"runtimes.{runtime_id}"
    inputs_value = mapping(value.get("inputs", {}), f"{context}.inputs")
    inputs: dict[str, str] = {}
    for input_name, sensor_id in inputs_value.items():
        if not input_name.strip():
            raise ConfigError(f"{context}.inputs keys must not be empty")
        if not isinstance(sensor_id, str) or not sensor_id.strip():
            raise ConfigError(
                f"{context}.inputs.{input_name} must reference a sensor ID"
            )
        inputs[input_name] = sensor_id
    return RuntimeConfig(
        runtime_id=runtime_id,
        robot=string(value, "robot", context),
        model=string(value, "model", context),
        binding=string(value, "binding", context),
        inputs=inputs,
        options={
            name: option
            for name, option in value.items()
            if name not in {"robot", "model", "binding", "inputs"}
        },
    )


__all__ = ["RuntimeConfig"]
