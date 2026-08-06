from __future__ import annotations

import math

import pytest

from embodied_runtime.utils import Pose2D, compose_relative_pose, normalize_angle, relative_pose


def test_relative_pose_round_trip() -> None:
    origin = Pose2D(1.2, -0.4, math.pi / 3)
    local = Pose2D(0.8, 0.2, -0.3)

    world = compose_relative_pose(origin, local)
    recovered = relative_pose(world, origin)

    assert recovered.x_m == pytest.approx(local.x_m)
    assert recovered.y_m == pytest.approx(local.y_m)
    assert recovered.yaw_rad == pytest.approx(local.yaw_rad)


def test_angles_are_normalized_and_finite() -> None:
    assert normalize_angle(3 * math.pi) == pytest.approx(math.pi)
    with pytest.raises(ValueError, match="finite"):
        Pose2D(0.0, float("nan"), 0.0)
