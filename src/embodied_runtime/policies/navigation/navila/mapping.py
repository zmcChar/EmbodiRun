"""Map one NaVILA native mid-level action into the navigation task contract."""

from __future__ import annotations

import math

from embodied_runtime.models.vla.navila import (
    MAX_FORWARD_CM,
    MAX_TURN_DEG,
    NaVILAAction,
    NaVILAPrediction,
    NaVILAPrimitive,
)
from embodied_runtime.tasks.navigation import Waypoint, WaypointPlan

MAX_NAVILA_FORWARD_M = MAX_FORWARD_CM / 100.0
MAX_NAVILA_TURN_RAD = math.radians(MAX_TURN_DEG)


def navila_prediction_to_waypoint_plan(
    prediction: NaVILAPrediction,
    observation_sequence: int,
) -> WaypointPlan:
    if not isinstance(prediction, NaVILAPrediction):
        raise TypeError("prediction must be a NaVILAPrediction")
    action: NaVILAAction = prediction.action
    if action.primitive is NaVILAPrimitive.STOP:
        return WaypointPlan(
            observation_sequence=observation_sequence,
            waypoints=(),
            terminal=True,
        )
    if action.magnitude is None:
        raise ValueError("NaVILA motion action has no magnitude")
    if action.primitive is NaVILAPrimitive.MOVE_FORWARD:
        waypoint = Waypoint(x_m=action.magnitude / 100.0, y_m=0.0, yaw_rad=0.0)
    else:
        direction = 1.0 if action.primitive is NaVILAPrimitive.TURN_LEFT else -1.0
        waypoint = Waypoint(
            x_m=0.0,
            y_m=0.0,
            yaw_rad=direction * math.radians(action.magnitude),
        )
    return WaypointPlan(
        observation_sequence=observation_sequence,
        waypoints=(waypoint,),
        terminal=False,
        confidence=1.0,
        valid_for_s=5.0,
    )


__all__ = [
    "MAX_NAVILA_FORWARD_M",
    "MAX_NAVILA_TURN_RAD",
    "navila_prediction_to_waypoint_plan",
]
