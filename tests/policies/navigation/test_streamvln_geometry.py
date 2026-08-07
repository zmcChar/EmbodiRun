from __future__ import annotations

import math

import pytest

from embodied_runtime.models.vln.streamvln import StreamVLNNativeOutputError
from embodied_runtime.policies.navigation.streamvln.geometry import (
    streamvln_actions_to_waypoint_plan,
)


def test_native_actions_form_cumulative_task_plan_and_stop_truncates() -> None:
    plan = streamvln_actions_to_waypoint_plan(
        [1, 2, 1, 0],
        observation_sequence=7,
    )

    assert plan.observation_sequence == 7
    assert plan.terminal
    assert len(plan.waypoints) == 3
    assert plan.waypoints[0].x_m == pytest.approx(0.25)
    assert plan.waypoints[0].y_m == pytest.approx(0.0)
    assert plan.waypoints[0].yaw_rad == pytest.approx(0.0)
    assert plan.waypoints[1].x_m == pytest.approx(0.25)
    assert plan.waypoints[1].yaw_rad == pytest.approx(math.pi / 12.0)
    assert plan.waypoints[2].x_m == pytest.approx(0.25 + 0.25 * math.cos(math.pi / 12.0))
    assert plan.waypoints[2].y_m == pytest.approx(0.25 * math.sin(math.pi / 12.0))

    stopped = streamvln_actions_to_waypoint_plan([0], observation_sequence=8)
    assert stopped.terminal
    assert stopped.waypoints == ()
    assert streamvln_actions_to_waypoint_plan([1, 0, 99], observation_sequence=9).terminal


@pytest.mark.parametrize("actions", [[], [True], [1.0], [4], [1, 1, 1, 1, 1]])
def test_invalid_native_actions_are_not_coerced(actions: list[object]) -> None:
    with pytest.raises(StreamVLNNativeOutputError):
        streamvln_actions_to_waypoint_plan(actions, observation_sequence=1)
