"""Robot instance configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation import optional_string, string


@dataclass(frozen=True, slots=True)
class RobotConfig:
    robot_id: str
    kind: str
    node: str
    port: str | None
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_robot(robot_id: str, value: dict[str, Any]) -> RobotConfig:
    context = f"robots.{robot_id}"
    return RobotConfig(
        robot_id=robot_id,
        kind=string(value, "type", context),
        node=string(value, "node", context),
        port=optional_string(value, "port", context),
        options=dict(value),
    )


__all__ = ["RobotConfig"]
