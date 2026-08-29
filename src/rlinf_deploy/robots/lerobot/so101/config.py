"""Configuration and fail-closed step limits for a LeRobot SO-101 follower."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SO101Config:
    port: str
    robot_id: str = "so101"
    calibration_id: str | None = None
    calibration_dir: str | None = None
    calibrate_on_connect: bool = True
    disable_torque_on_disconnect: bool = True
    max_joint_step_deg: float = 12.0
    max_gripper_step: float = 20.0

    def __post_init__(self) -> None:
        if not self.port.strip():
            raise ValueError("port must not be empty")
        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if self.calibration_id is not None and not self.calibration_id.strip():
            raise ValueError("calibration_id must not be empty")
        if self.calibration_dir is not None and not self.calibration_dir.strip():
            raise ValueError("calibration_dir must not be empty")
        if self.max_joint_step_deg <= 0:
            raise ValueError("max_joint_step_deg must be positive")
        if self.max_gripper_step <= 0:
            raise ValueError("max_gripper_step must be positive")


__all__ = ["SO101Config"]
