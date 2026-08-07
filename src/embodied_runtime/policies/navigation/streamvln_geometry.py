"""Interpret StreamVLN native actions as task-owned base-frame waypoints."""

from __future__ import annotations

import math
from collections.abc import Sequence

from embodied_runtime.models.vln.streamvln import (
    MAX_FUTURE_ACTIONS,
    StreamVLNAction,
    normalize_native_actions,
)
from embodied_runtime.tasks.navigation import Waypoint, WaypointPlan

FORWARD_STEP_M = 0.25
TURN_STEP_RAD = math.radians(15.0)


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def streamvln_actions_to_waypoint_plan(
    actions: Sequence[object],
    *,
    observation_sequence: int,
    forward_step_m: float = FORWARD_STEP_M,
    turn_step_rad: float = TURN_STEP_RAD,
    max_actions: int = MAX_FUTURE_ACTIONS,
) -> WaypointPlan:
    """Convert native actions into cumulative capture-time task waypoints.

    Turns intentionally produce waypoints so a controller completes the
    requested heading before consuming the following forward target. A bare
    STOP is therefore the only valid empty plan.
    """

    forward_step = _positive_finite(forward_step_m, "forward_step_m")
    turn_step = _positive_finite(turn_step_rad, "turn_step_rad")
    normalized = normalize_native_actions(actions, max_actions=max_actions)

    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0
    terminal = False
    waypoints: list[Waypoint] = []
    for action_id in normalized:
        action = StreamVLNAction(action_id)
        if action is StreamVLNAction.STOP:
            terminal = True
            break
        if action is StreamVLNAction.FORWARD:
            x_m += forward_step * math.cos(yaw_rad)
            y_m += forward_step * math.sin(yaw_rad)
        elif action is StreamVLNAction.TURN_LEFT:
            yaw_rad = _normalize_angle(yaw_rad + turn_step)
        elif action is StreamVLNAction.TURN_RIGHT:
            yaw_rad = _normalize_angle(yaw_rad - turn_step)
        waypoints.append(Waypoint(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad))

    return WaypointPlan(
        observation_sequence=observation_sequence,
        waypoints=tuple(waypoints),
        terminal=terminal,
        confidence=1.0,
        valid_for_s=5.0,
    )


__all__ = [
    "FORWARD_STEP_M",
    "TURN_STEP_RAD",
    "streamvln_actions_to_waypoint_plan",
]
