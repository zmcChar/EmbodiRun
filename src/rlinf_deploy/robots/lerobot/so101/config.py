"""Configuration and fail-closed step limits for a LeRobot SO-101 follower."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SO101PositionMode = Literal["degrees", "normalized"]
SO101StepLimitMode = Literal["reject", "clip"]


@dataclass(frozen=True, slots=True)
class SO101Config:
    port: str
    robot_id: str = "so101"
    calibration_id: str | None = None
    calibration_dir: Path | None = None
    disable_torque_on_disconnect: bool = True
    position_mode: SO101PositionMode = "degrees"
    step_limit_mode: SO101StepLimitMode = "reject"
    max_joint_step_deg: float = 12.0
    max_joint_step_normalized: float = 12.0
    max_gripper_step: float = 20.0

    def __post_init__(self) -> None:
        if not self.port.strip():
            raise ValueError("port must not be empty")
        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if self.calibration_id is not None and not self.calibration_id.strip():
            raise ValueError("calibration_id must not be empty")
        if self.position_mode not in {"degrees", "normalized"}:
            raise ValueError("position_mode must be 'degrees' or 'normalized'")
        if self.step_limit_mode not in {"reject", "clip"}:
            raise ValueError("step_limit_mode must be 'reject' or 'clip'")
        if not math.isfinite(self.max_joint_step_deg) or self.max_joint_step_deg <= 0:
            raise ValueError("max_joint_step_deg must be positive")
        if (
            not math.isfinite(self.max_joint_step_normalized)
            or self.max_joint_step_normalized <= 0
        ):
            raise ValueError("max_joint_step_normalized must be positive")
        if not math.isfinite(self.max_gripper_step) or self.max_gripper_step <= 0:
            raise ValueError("max_gripper_step must be positive")


__all__ = ["SO101Config", "SO101PositionMode", "SO101StepLimitMode"]
