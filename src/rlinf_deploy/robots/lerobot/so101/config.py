"""Configuration and fail-closed step limits for a LeRobot SO-101 follower."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StepLimitMode = Literal["reject", "clip"]


@dataclass(frozen=True, slots=True)
class SO101Config:
    """Connection, calibration, and motion-safety settings for one SO-101."""

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
        for name in ("max_joint_step_deg", "max_gripper_step"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.step_limit_mode, str) or self.step_limit_mode not in {
            "reject",
            "clip",
        }:
            raise ValueError("step_limit_mode must be 'reject' or 'clip'")


__all__ = ["SO101Config", "StepLimitMode"]
