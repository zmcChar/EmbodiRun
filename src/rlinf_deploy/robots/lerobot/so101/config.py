"""Configuration and fail-closed step limits for a LeRobot SO-101 follower."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

StepLimitMode = Literal["reject", "clip"]


@dataclass(frozen=True, slots=True)
class SO101Config:
    port: str
    robot_id: str = "so101"
    calibration_id: str | None = None
    calibration_dir: Path | None = None
    disable_torque_on_disconnect: bool = True
    max_joint_step_deg: float = 12.0
    max_gripper_step: float = 20.0
    step_limit_mode: StepLimitMode = "reject"

    def __post_init__(self) -> None:
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("port must not be empty")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if self.calibration_id is not None and (
            not isinstance(self.calibration_id, str) or not self.calibration_id.strip()
        ):
            raise ValueError("calibration_id must not be empty")
        if self.calibration_dir is not None and not isinstance(
            self.calibration_dir, Path
        ):
            raise TypeError("calibration_dir must be a Path or None")
        if not isinstance(self.disable_torque_on_disconnect, bool):
            raise TypeError("disable_torque_on_disconnect must be a boolean")
        if (
            isinstance(self.max_joint_step_deg, bool)
            or not isinstance(self.max_joint_step_deg, (int, float))
            or not math.isfinite(self.max_joint_step_deg)
            or self.max_joint_step_deg <= 0
        ):
            raise ValueError("max_joint_step_deg must be positive")
        if (
            isinstance(self.max_gripper_step, bool)
            or not isinstance(self.max_gripper_step, (int, float))
            or not math.isfinite(self.max_gripper_step)
            or self.max_gripper_step <= 0
        ):
            raise ValueError("max_gripper_step must be positive")
        if self.step_limit_mode not in {"reject", "clip"}:
            raise ValueError("step_limit_mode must be 'reject' or 'clip'")

    @classmethod
    def from_yaml(cls, robot_id: str, value: Mapping[str, Any]) -> SO101Config:
        """Parse one decoded ``robots.<id>`` YAML mapping."""

        context = f"robots.{robot_id}"
        _reject_unknown(
            value,
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
                "sensors",
            },
            context,
        )
        if value.get("type") != "lerobot.so101":
            raise ValueError(f"{context}.type must be 'lerobot.so101'")
        return cls._from_mapping(value, robot_id=robot_id, context=context)

    @classmethod
    def from_runtime_payload(cls, value: Mapping[str, Any]) -> SO101Config:
        """Parse SO101 fields from a binding runtime payload."""

        context = "runtime"
        return cls._from_mapping(
            value,
            robot_id=_string(value, "robot_id", context),
            context=context,
            port_key="robot_port",
        )

    @classmethod
    def _from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        robot_id: str,
        context: str,
        port_key: str = "port",
    ) -> SO101Config:
        return cls(
            port=_string(value, port_key, context),
            robot_id=robot_id,
            calibration_id=_optional_string(value, "calibration_id", context),
            calibration_dir=_optional_path(value, "calibration_dir", context),
            disable_torque_on_disconnect=_boolean(
                value,
                "disable_torque_on_disconnect",
                context,
                default=True,
            ),
            max_joint_step_deg=_positive_number(
                value,
                "max_joint_step_deg",
                context,
                default=12.0,
            ),
            max_gripper_step=_positive_number(
                value,
                "max_gripper_step",
                context,
                default=20.0,
            ),
            step_limit_mode=_step_limit_mode(value, context),
        )

    def to_runtime_payload(self) -> dict[str, object]:
        """Return the SO101 fields in the version-one runtime wire format."""

        return {
            "robot_port": self.port,
            "robot_id": self.robot_id,
            "calibration_id": self.calibration_id,
            "calibration_dir": (
                str(self.calibration_dir) if self.calibration_dir is not None else None
            ),
            "disable_torque_on_disconnect": self.disable_torque_on_disconnect,
            "max_joint_step_deg": self.max_joint_step_deg,
            "max_gripper_step": self.max_gripper_step,
            "step_limit_mode": self.step_limit_mode,
        }


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _string(value: Mapping[str, Any], name: str, context: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context}.{name} must be a non-empty string")
    return result


def _optional_string(
    value: Mapping[str, Any],
    name: str,
    context: str,
) -> str | None:
    result = value.get(name)
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context}.{name} must be a non-empty string when provided")
    return result


def _optional_path(
    value: Mapping[str, Any],
    name: str,
    context: str,
) -> Path | None:
    result = _optional_string(value, name, context)
    return Path(result) if result is not None else None


def _boolean(
    value: Mapping[str, Any],
    name: str,
    context: str,
    *,
    default: bool,
) -> bool:
    result = value.get(name, default)
    if not isinstance(result, bool):
        raise TypeError(f"{context}.{name} must be a boolean")
    return result


def _positive_number(
    value: Mapping[str, Any],
    name: str,
    context: str,
    *,
    default: float,
) -> float:
    result = value.get(name, default)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise TypeError(f"{context}.{name} must be a positive number")
    number = float(result)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{context}.{name} must be a positive number")
    return number


def _step_limit_mode(value: Mapping[str, Any], context: str) -> StepLimitMode:
    result = value.get("step_limit_mode", "reject")
    if not isinstance(result, str) or result not in {"reject", "clip"}:
        raise ValueError(f"{context}.step_limit_mode must be 'reject' or 'clip'")
    return cast(StepLimitMode, result)


__all__ = ["SO101Config", "StepLimitMode"]
