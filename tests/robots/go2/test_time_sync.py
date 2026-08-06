from __future__ import annotations

import math

import pytest

from embodied_runtime.robots.go2 import Go2State, capture_pose, project_pose_to_time
from embodied_runtime.utils import Pose2D


def _state(timestamp: float | None, *, x_m: float = 0.0) -> Go2State:
    return Go2State(
        pose=Pose2D(x_m, 0.0, math.pi / 2),
        forward_velocity_mps=1.0,
        lateral_velocity_mps=0.0,
        yaw_rate_rps=0.0,
        sequence=1,
        received_at_s=timestamp,
    )


def test_pose_projection_rotates_body_velocity_into_odometry() -> None:
    projected = project_pose_to_time(_state(10.0), 10.2)
    assert projected.x_m == pytest.approx(0.0, abs=1e-9)
    assert projected.y_m == pytest.approx(0.2)


def test_capture_pose_uses_nearest_timestamp_and_bounds_extrapolation() -> None:
    before = _state(10.0, x_m=1.0)
    after = _state(10.2, x_m=1.1)
    result = capture_pose(before, after, 10.18)
    assert result.x_m == pytest.approx(1.1)
    assert result.y_m == pytest.approx(-0.02)
    assert project_pose_to_time(after, 20.0) == after.pose
