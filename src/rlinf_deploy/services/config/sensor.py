"""Sensor instance configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation import string


@dataclass(frozen=True, slots=True)
class SensorConfig:
    sensor_id: str
    kind: str
    node: str
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_sensor(sensor_id: str, value: dict[str, Any]) -> SensorConfig:
    context = f"sensors.{sensor_id}"
    return SensorConfig(
        sensor_id=sensor_id,
        kind=string(value, "type", context),
        node=string(value, "node", context),
        options={
            name: option
            for name, option in value.items()
            if name not in {"type", "node"}
        },
    )


__all__ = ["SensorConfig"]
