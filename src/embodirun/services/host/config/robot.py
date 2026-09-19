"""Robot instance configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodirun.robots import robot_definition

from .validation import (
    ConfigError,
    optional_string,
    string,
)


@dataclass(frozen=True, slots=True)
class RobotConfig:
    robot_id: str
    kind: str
    node: str
    port: str | None
    options: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ports(self) -> tuple[str, ...]:
        """Serial devices claimed by this robot, including composite arms."""
        return tuple(
            self.options[name]
            for name in robot_definition(self.kind).port_fields
            if self.options.get(name) is not None
        )


def parse_robot(robot_id: str, value: dict[str, Any]) -> RobotConfig:
    context = f"robots.{robot_id}"
    kind = string(value, "type", context)
    options = {
        name: option
        for name, option in value.items()
        if name not in {"type", "node"}
    }
    try:
        definition = robot_definition(kind)
    except (KeyError, TypeError):
        raise ConfigError(f"{context}.type {kind!r} is not supported") from None
    try:
        definition.config_factory(robot_id, options)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{context}: {error}") from error
    return RobotConfig(
        robot_id=robot_id,
        kind=kind,
        node=string(value, "node", context),
        port=optional_string(value, "port", context),
        options=options,
    )


__all__ = ["RobotConfig", "parse_robot"]
