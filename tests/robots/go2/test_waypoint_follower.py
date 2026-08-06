from __future__ import annotations

import math

import pytest

from embodied_runtime.contracts import Waypoint, WaypointPlan
from embodied_runtime.robots.go2 import WorldWaypointFollower
from embodied_runtime.utils import Pose2D


def test_plan_is_anchored_at_capture_pose_and_recomputed_at_control_rate() -> None:
    follower = WorldWaypointFollower()
    follower.replace(
        WaypointPlan(10, (Waypoint(1.0, 0.0, 0.0),), valid_for_s=2.0),
        Pose2D(2.0, 3.0, math.pi / 2),
        accepted_at_s=5.0,
    )

    first = follower.sample(Pose2D(2.0, 3.0, math.pi / 2), 5.1)
    later = follower.sample(Pose2D(2.0, 3.7, math.pi / 2), 5.5)

    assert first.active and first.command.vx_mps > 0
    assert later.active and 0 < later.command.vx_mps < first.command.vx_mps


def test_terminal_plan_stops_after_last_waypoint() -> None:
    follower = WorldWaypointFollower()
    follower.replace(
        WaypointPlan(1, (Waypoint(0.1, 0.0, 0.0),), terminal=True),
        Pose2D(0.0, 0.0, 0.0),
        accepted_at_s=1.0,
    )
    sample = follower.sample(Pose2D(0.1, 0.0, 0.0), 1.1)
    assert sample.terminal_reached
    assert not sample.command.moving


def test_expired_plan_is_dropped() -> None:
    follower = WorldWaypointFollower()
    follower.replace(
        WaypointPlan(1, (Waypoint(1.0, 0.0, 0.0),), valid_for_s=0.2),
        Pose2D(0.0, 0.0, 0.0),
        accepted_at_s=1.0,
    )
    assert not follower.sample(Pose2D(0.0, 0.0, 0.0), 1.3).active

    with pytest.raises(ValueError, match="increase"):
        follower.replace(
            WaypointPlan(0, (Waypoint(1.0, 0.0, 0.0),)),
            Pose2D(0.0, 0.0, 0.0),
            accepted_at_s=2.0,
        )
