"""Robot instance configuration and parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from rlinf_deploy.robots.lerobot.so101 import SO101Config, StepLimitMode

from .validation import (
    ConfigError,
    boolean,
    optional_string,
    positive_number,
    reject_unknown,
    string,
)


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


def parse_so101_config(
    robot_id: str,
    robot_kind: str,
    robot_options: Mapping[str, Any],
) -> SO101Config:
    """Build the typed SO-101 configuration from one YAML robot entry."""

    context = f"robots.{robot_id}"
    if robot_kind != "lerobot.so101":
        raise ConfigError(f"{context}.type must be 'lerobot.so101'")
    options = dict(robot_options)
    reject_unknown(
        options,
        {
            "type",
            "node",
            "port",
            "calibration_id",
            "calibration_dir",
            "disable_torque_on_disconnect",
            "max_joint_step_deg",
            "max_gripper_step",
            "step_limit_mode",
        },
        context,
    )
    step_limit_mode_value = options.get("step_limit_mode", "reject")
    if not isinstance(step_limit_mode_value, str) or step_limit_mode_value not in {
        "reject",
        "clip",
    }:
        raise ConfigError(f"{context}.step_limit_mode must be 'reject' or 'clip'")
    step_limit_mode = cast(StepLimitMode, step_limit_mode_value)
    calibration_dir = optional_string(options, "calibration_dir", context)
    try:
        return SO101Config(
            port=string(options, "port", context),
            robot_id=robot_id,
            calibration_id=optional_string(options, "calibration_id", context),
            calibration_dir=Path(calibration_dir) if calibration_dir else None,
            disable_torque_on_disconnect=boolean(
                options.get("disable_torque_on_disconnect", True),
                f"{context}.disable_torque_on_disconnect",
            ),
            max_joint_step_deg=positive_number(
                options.get("max_joint_step_deg", 12.0),
                f"{context}.max_joint_step_deg",
            ),
            max_gripper_step=positive_number(
                options.get("max_gripper_step", 20.0),
                f"{context}.max_gripper_step",
            ),
            step_limit_mode=step_limit_mode,
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{context}: {error}") from error


__all__ = ["RobotConfig", "parse_so101_config"]
