"""Safety limits for Franky motion commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FR3Config:
    host: str
    robot_id: str = "fr3"
    relative_dynamics_factor: float = 0.05
    max_joint_step_rad: float = 0.20
    max_cartesian_step_m: float = 0.05
    enable_gripper: bool = True
    gripper_speed_mps: float = 0.02
    max_gripper_width_m: float = 0.08

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host must not be empty")
        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if not 0 < self.relative_dynamics_factor <= 1:
            raise ValueError("relative_dynamics_factor must be in (0, 1]")
        for name in (
            "max_joint_step_rad",
            "max_cartesian_step_m",
            "gripper_speed_mps",
            "max_gripper_width_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
