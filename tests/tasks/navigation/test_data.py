from __future__ import annotations

import pytest

from embodied_runtime.tasks.navigation import (
    EncodedDepthFrame,
    EncodedRGBFrame,
    NavigationContractError,
    NavigationObservation,
    NavigationRequest,
    Waypoint,
    WaypointPlan,
)


def rgb(sequence: int = 7, captured_at_s: float = 10.0) -> EncodedRGBFrame:
    return EncodedRGBFrame(
        sequence=sequence,
        captured_at_s=captured_at_s,
        data=b"\xff\xd8jpeg",
        width=640,
        height=360,
    )


def depth(sequence: int = 7, captured_at_s: float = 10.0) -> EncodedDepthFrame:
    return EncodedDepthFrame(
        sequence=sequence,
        captured_at_s=captured_at_s,
        data=b"\x89PNG\r\n\x1a\nraw-z16",
        width=640,
        height=360,
        scale_m=0.001,
    )


def test_atomic_rgbd_observation_preserves_episode_lineage() -> None:
    observation = NavigationObservation(
        episode_id="go2:tripod-1",
        sequence=7,
        reset=True,
        rgb_frames=(rgb(),),
        depth=depth(),
        robot_state={"position": [1.0, 2.0], "yaw": 0.2},
    )
    request = NavigationRequest("导航到三脚架前", observation)

    assert request.instruction == "导航到三脚架前"
    assert request.observation.latest_rgb.sequence == 7
    assert request.observation.depth is not None
    assert request.observation.depth.scale_m == pytest.approx(0.001)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sequence": 8}, "latest RGB sequence"),
        ({"depth": depth(sequence=8)}, "depth sequence"),
        ({"depth": depth(captured_at_s=10.1)}, "capture time"),
    ],
)
def test_observation_rejects_mismatched_sensor_lineage(kwargs, message: str) -> None:
    values = {
        "episode_id": "episode-1",
        "sequence": 7,
        "reset": False,
        "rgb_frames": (rgb(),),
        "depth": depth(),
    }
    values.update(kwargs)
    with pytest.raises(NavigationContractError, match=message):
        NavigationObservation(**values)


def test_stop_is_terminal_instead_of_a_fake_zero_waypoint() -> None:
    plan = WaypointPlan(observation_sequence=7, waypoints=(), terminal=True)
    assert plan.terminal
    assert plan.waypoints == ()

    with pytest.raises(NavigationContractError, match="empty waypoint plan"):
        WaypointPlan(observation_sequence=7, waypoints=(), terminal=False)


def test_waypoint_plan_retains_spatial_semantics() -> None:
    plan = WaypointPlan(
        observation_sequence=7,
        waypoints=(Waypoint(0.8, 0.1, 0.2),),
        confidence=0.9,
        valid_for_s=1.5,
    )
    assert plan.frame == "base_link"
    assert plan.waypoints[0].x_m == pytest.approx(0.8)


def test_encoded_frames_reject_bad_signatures_and_unregistered_depth() -> None:
    with pytest.raises(NavigationContractError, match="file signature"):
        EncodedRGBFrame(sequence=1, captured_at_s=1.0, data=b"not-jpeg")
    with pytest.raises(NavigationContractError, match="registered"):
        EncodedDepthFrame(
            sequence=1,
            captured_at_s=1.0,
            data=b"\x89PNG\r\n\x1a\nraw",
            width=1,
            height=1,
            scale_m=0.001,
            registered_to_rgb=False,
        )
