from __future__ import annotations

import math

import pytest

from embodied_runtime.tasks.navigation import (
    MobileBaseState,
    PlanarVelocityCommand,
    ReactiveNavigationSessionConfig,
    Waypoint,
    WaypointPlan,
)
from embodied_runtime.tasks.navigation.controllers import (
    VelocityPulseConfig,
    WorldWaypointFollower,
    capture_pose,
    project_pose_to_time,
    waypoint_to_velocity_pulse,
)
from embodied_runtime.utils import Pose2D


def _state(timestamp: float | None, *, x_m: float = 0.0) -> MobileBaseState:
    return MobileBaseState(
        Pose2D(x_m, 0, math.pi / 2),
        1.0,
        0.0,
        0.0,
        1,
        timestamp,
    )


def test_pose_projection_and_nearest_capture_state() -> None:
    projected = project_pose_to_time(_state(10.0), 10.2)
    assert projected.x_m == pytest.approx(0.0, abs=1e-9)
    assert projected.y_m == pytest.approx(0.2)

    before = _state(10.0, x_m=1.0)
    after = _state(10.2, x_m=1.1)
    captured = capture_pose(before, after, 10.18)
    assert captured.x_m == pytest.approx(1.1)
    assert captured.y_m == pytest.approx(-0.02)


def test_follower_anchors_plan_and_rejects_old_sequence() -> None:
    follower = WorldWaypointFollower()
    follower.replace(
        WaypointPlan(10, (Waypoint(1, 0, 0),), valid_for_s=2),
        Pose2D(2, 3, math.pi / 2),
        5.0,
    )

    first = follower.sample(Pose2D(2, 3, math.pi / 2), 5.1)
    later = follower.sample(Pose2D(2, 3.7, math.pi / 2), 5.5)
    assert first.active and first.command.vx_mps > 0
    assert later.active and 0 < later.command.vx_mps < first.command.vx_mps
    with pytest.raises(ValueError, match="increase"):
        follower.replace(WaypointPlan(9, (Waypoint(1, 0, 0),)), Pose2D(0, 0, 0), 6)


def test_terminal_and_expired_follower_samples_are_stopped() -> None:
    follower = WorldWaypointFollower()
    follower.replace(
        WaypointPlan(1, (Waypoint(0.1, 0, 0),), terminal=True),
        Pose2D(0, 0, 0),
        1.0,
    )
    terminal = follower.sample(Pose2D(0.1, 0, 0), 1.1)
    assert terminal.terminal_reached and not terminal.command.moving

    follower.reset_episode()
    follower.replace(
        WaypointPlan(2, (Waypoint(1, 0, 0),), valid_for_s=0.2),
        Pose2D(0, 0, 0),
        2.0,
    )
    assert not follower.sample(Pose2D(0, 0, 0), 2.3).active


def test_streamvln_sized_forward_and_turn_become_bounded_pulses() -> None:
    forward, forward_s = waypoint_to_velocity_pulse(Waypoint(0.25, 0, 0))
    turn, turn_s = waypoint_to_velocity_pulse(Waypoint(0, 0, -math.radians(15)))

    assert isinstance(forward, PlanarVelocityCommand)
    assert forward.vx_mps == pytest.approx(0.30)
    assert forward_s == pytest.approx(0.25 / 0.30)
    assert turn.yaw_rate_rps == pytest.approx(-0.60)
    assert turn_s == pytest.approx(math.radians(15) / 0.60)
    with pytest.raises(ValueError, match="min_duration_s"):
        VelocityPulseConfig(min_duration_s=2, max_duration_s=1)


def test_reactive_config_remains_separate_from_pulse_algorithm_config() -> None:
    config = ReactiveNavigationSessionConfig()
    assert config.max_pulse_s == 1.5
