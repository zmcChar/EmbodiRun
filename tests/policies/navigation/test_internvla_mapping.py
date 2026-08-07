from __future__ import annotations

import math

import numpy as np
import pytest

from embodied_runtime.models.vla.internvla_n1 import (
    InternVLAOutputError,
    NativePrediction,
)
from embodied_runtime.policies.navigation.internvla_mapping import (
    internvla_prediction_to_waypoint_plan,
)


def test_trajectory_drops_warmup_prefix_and_uses_path_tangents() -> None:
    plan = internvla_prediction_to_waypoint_plan(
        NativePrediction(
            trajectory=[
                [0.0, 0.0],
                [0.02, 0.0],
                [0.05, 0.01],
                [0.10, 0.02],
                [0.30, 0.10],
            ]
        ),
        7,
    )

    assert not plan.terminal
    assert [(point.x_m, point.y_m) for point in plan.waypoints] == [
        (0.1, 0.02),
        (0.3, 0.1),
    ]
    expected_yaw = math.atan2(0.08, 0.2)
    assert plan.waypoints[0].yaw_rad == pytest.approx(expected_yaw)
    assert plan.waypoints[1].yaw_rad == pytest.approx(expected_yaw)


def test_discrete_actions_convert_to_base_frame_task_waypoints() -> None:
    plan = internvla_prediction_to_waypoint_plan(
        NativePrediction(discrete_action=np.array([1, 2, 1, 3])),
        2,
    )

    assert len(plan.waypoints) == 4
    assert plan.waypoints[0].x_m == pytest.approx(0.25)
    assert plan.waypoints[1].yaw_rad == pytest.approx(math.radians(15.0))
    assert plan.waypoints[2].x_m == pytest.approx(0.25 + 0.25 * math.cos(math.radians(15.0)))
    assert plan.waypoints[2].y_m > 0.0
    assert plan.waypoints[-1].yaw_rad == pytest.approx(0.0)


def test_stop_is_terminal_and_malformed_native_unions_are_rejected() -> None:
    stopped = internvla_prediction_to_waypoint_plan(
        NativePrediction(discrete_action=[0]),
        9,
    )
    assert stopped.terminal
    assert stopped.waypoints == ()

    invalid = (
        NativePrediction(),
        NativePrediction(trajectory=[[0, 0]] * 4, discrete_action=[1]),
        NativePrediction(trajectory=[[0, 0]] * 3),
        NativePrediction(trajectory=[[0, 0]] * 3 + [[True, 0]]),
        NativePrediction(trajectory=[[0, 0]] * 3 + [["1", 0]]),
        NativePrediction(discrete_action=[0, 1]),
        NativePrediction(discrete_action=[5]),
        NativePrediction(discrete_action=[9]),
    )
    for native in invalid:
        with pytest.raises(InternVLAOutputError):
            internvla_prediction_to_waypoint_plan(native, 0)
