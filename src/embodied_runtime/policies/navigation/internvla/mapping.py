"""Map InternVLA native trajectory/action outputs into task waypoint plans."""

from __future__ import annotations

import math
from itertools import pairwise
from numbers import Real
from typing import Any

from embodied_runtime.models.vla.internvla_n1 import (
    InternVLAOutputError,
    NativePrediction,
    normalize_discrete_actions,
)
from embodied_runtime.tasks.navigation import Waypoint, WaypointPlan

MAX_WAYPOINTS = 64
MAX_WAYPOINT_COORDINATE_M = 20.0
TRAJECTORY_PREFIX_POINTS_TO_DROP = 3
FORWARD_STEP_M = 0.25
TURN_STEP_RAD = math.radians(15.0)
DEFAULT_VALID_FOR_S = 5.0


def _as_list(value: object, name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()  # type: ignore[union-attr]
        except Exception as error:
            raise InternVLAOutputError(f"cannot convert {name} to a list: {error}") from error
    if not isinstance(value, (list, tuple)):
        raise InternVLAOutputError(f"{name} must be a sequence")
    return list(value)


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InternVLAOutputError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise InternVLAOutputError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise InternVLAOutputError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise InternVLAOutputError(f"{name} must be at most {maximum}")
    return result


def _normalize_angle(value: float) -> float:
    normalized = (value + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(normalized, -math.pi) and value > 0.0:
        return math.pi
    return normalized


def _trajectory_waypoints(value: object) -> tuple[Waypoint, ...]:
    points = _as_list(value, "trajectory")
    if len(points) <= TRAJECTORY_PREFIX_POINTS_TO_DROP:
        raise InternVLAOutputError("trajectory is too short after dropping warm-up points")
    if len(points) - TRAJECTORY_PREFIX_POINTS_TO_DROP > MAX_WAYPOINTS:
        raise InternVLAOutputError("trajectory exceeds the supported waypoint horizon")

    coordinates: list[tuple[float, float]] = []
    for index, point_value in enumerate(
        points[TRAJECTORY_PREFIX_POINTS_TO_DROP:],
        start=TRAJECTORY_PREFIX_POINTS_TO_DROP,
    ):
        point = _as_list(point_value, f"trajectory[{index}]")
        if len(point) != 2:
            raise InternVLAOutputError(f"trajectory[{index}] must contain [x, y]")
        coordinates.append(
            (
                _finite(
                    point[0],
                    f"trajectory[{index}][0]",
                    minimum=-MAX_WAYPOINT_COORDINATE_M,
                    maximum=MAX_WAYPOINT_COORDINATE_M,
                ),
                _finite(
                    point[1],
                    f"trajectory[{index}][1]",
                    minimum=-MAX_WAYPOINT_COORDINATE_M,
                    maximum=MAX_WAYPOINT_COORDINATE_M,
                ),
            )
        )

    segment_headings: list[float | None] = []
    for current, following in pairwise(coordinates):
        delta_x = following[0] - current[0]
        delta_y = following[1] - current[1]
        segment_headings.append(
            None if math.hypot(delta_x, delta_y) <= 1e-9 else math.atan2(delta_y, delta_x)
        )
    latest_heading = next(
        (heading for heading in segment_headings if heading is not None),
        0.0,
    )
    waypoints: list[Waypoint] = []
    for index, (x_m, y_m) in enumerate(coordinates):
        candidate = segment_headings[index] if index < len(segment_headings) else None
        if candidate is not None:
            latest_heading = candidate
        waypoints.append(Waypoint(x_m=x_m, y_m=y_m, yaw_rad=latest_heading))
    return tuple(waypoints)


def _discrete_waypoints(actions: list[int]) -> tuple[Waypoint, ...]:
    if 0 in actions:
        raise InternVLAOutputError("STOP may not be mixed with motion actions")
    if 5 in actions:
        raise InternVLAOutputError("look-down action escaped the runtime's bounded retry")
    unknown = sorted(set(actions) - {1, 2, 3})
    if unknown:
        raise InternVLAOutputError(f"unsupported discrete actions: {unknown}")

    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0
    waypoints: list[Waypoint] = []
    for action in actions:
        if action == 1:
            x_m += FORWARD_STEP_M * math.cos(yaw_rad)
            y_m += FORWARD_STEP_M * math.sin(yaw_rad)
        elif action == 2:
            yaw_rad = _normalize_angle(yaw_rad + TURN_STEP_RAD)
        else:
            yaw_rad = _normalize_angle(yaw_rad - TURN_STEP_RAD)
        waypoints.append(Waypoint(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad))
    return tuple(waypoints)


def internvla_prediction_to_waypoint_plan(
    native: NativePrediction,
    observation_sequence: int,
    *,
    valid_for_s: float = DEFAULT_VALID_FOR_S,
) -> WaypointPlan:
    """Interpret exactly one native output arm as a task-owned plan."""

    if isinstance(observation_sequence, bool) or not isinstance(observation_sequence, int):
        raise InternVLAOutputError("observation_sequence must be a non-negative integer")
    if observation_sequence < 0:
        raise InternVLAOutputError("observation_sequence must be a non-negative integer")
    lifetime = _finite(
        valid_for_s,
        "valid_for_s",
        minimum=math.nextafter(0.0, 1.0),
        maximum=10.0,
    )
    present = int(native.trajectory is not None) + int(native.discrete_action is not None)
    if present != 1:
        raise InternVLAOutputError(
            "InternVLA output must contain exactly one of trajectory or discrete_action"
        )
    if native.trajectory is not None:
        waypoints = _trajectory_waypoints(native.trajectory)
        terminal = False
    else:
        actions = normalize_discrete_actions(native.discrete_action)
        terminal = actions == [0]
        waypoints = () if terminal else _discrete_waypoints(actions)
    return WaypointPlan(
        observation_sequence=observation_sequence,
        waypoints=waypoints,
        terminal=terminal,
        valid_for_s=lifetime,
        confidence=1.0,
    )


__all__ = [
    "DEFAULT_VALID_FOR_S",
    "FORWARD_STEP_M",
    "MAX_WAYPOINTS",
    "TRAJECTORY_PREFIX_POINTS_TO_DROP",
    "TURN_STEP_RAD",
    "internvla_prediction_to_waypoint_plan",
]
