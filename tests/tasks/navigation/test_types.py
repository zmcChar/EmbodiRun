from __future__ import annotations

import math

import pytest

from embodied_runtime.tasks.navigation import (
    EncodedDepthFrame,
    EncodedRGBFrame,
    MobileBaseState,
    NavigationContractError,
    NavigationObservation,
    NavigationRequest,
    PlanarVelocityCommand,
    PlanarVelocityLimits,
    Waypoint,
    WaypointPlan,
)
from embodied_runtime.tasks.navigation.frames import EncodedRGBFrame as CanonicalRGB
from embodied_runtime.tasks.navigation.motion import MobileBaseState as CanonicalState
from embodied_runtime.tasks.navigation.plan import WaypointPlan as CanonicalPlan
from embodied_runtime.utils import Pose2D

_JPEG = b"\xff\xd8\xff\xd9"
_PNG = b"\x89PNG\r\n\x1a\nrest"


def test_public_api_reexports_canonical_types() -> None:
    assert EncodedRGBFrame is CanonicalRGB
    assert MobileBaseState is CanonicalState
    assert WaypointPlan is CanonicalPlan


def test_rgbd_observation_preserves_lineage_and_request() -> None:
    rgb = EncodedRGBFrame(7, 123.5, _JPEG, width=2, height=1)
    depth = EncodedDepthFrame(7, 123.5, _PNG, 2, 1, 0.001)
    observation = NavigationObservation(
        "episode:7",
        7,
        True,
        (rgb,),
        depth,
        robot_state={"yaw": 0.1},
    )
    request = NavigationRequest("find the yellow post", observation)

    assert observation.latest_rgb is rgb
    assert request.observation is observation
    assert observation.depth is depth


def test_observation_rejects_rgb_depth_mismatch() -> None:
    rgb = EncodedRGBFrame(2, 10.0, _JPEG, width=2, height=1)
    depth = EncodedDepthFrame(2, 10.1, _PNG, 2, 1, 0.001)
    with pytest.raises(NavigationContractError, match="capture time"):
        NavigationObservation("episode", 2, False, (rgb,), depth)


def test_waypoint_plan_requires_terminal_when_empty() -> None:
    assert WaypointPlan(3, (), terminal=True).terminal
    with pytest.raises(NavigationContractError, match="empty waypoint"):
        WaypointPlan(3, ())
    with pytest.raises(NavigationContractError, match="yaw_rad"):
        Waypoint(0, 0, math.pi + 0.01)


def test_planar_velocity_and_state_are_validated_and_serializable() -> None:
    limits = PlanarVelocityLimits(0.2, 0.1, 0.4)
    command = PlanarVelocityCommand(0.2, -0.1, 0.4, limits)
    state = MobileBaseState(
        Pose2D(1, 2, 0.3),
        0.2,
        -0.1,
        0.4,
        8,
        100.0,
        active_velocity_lease_id="lease-1",
        raw={"battery": 0.9},
    )

    assert command.moving
    assert not PlanarVelocityCommand.stopped(limits).moving
    assert state.as_observation_metadata() == {
        "battery": 0.9,
        "position": [1.0, 2.0],
        "yaw": 0.3,
        "velocity": [0.2, -0.1],
        "yaw_rate": 0.4,
        "sequence": 8,
        "received_at_unix": 100.0,
        "active_velocity_lease_id": "lease-1",
    }
    with pytest.raises(ValueError, match="vx_mps"):
        PlanarVelocityCommand(0.21, 0, 0, limits)
