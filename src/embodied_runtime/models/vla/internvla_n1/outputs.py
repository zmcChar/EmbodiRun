"""InternVLA native-output validation and spatial waypoint conversion."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from typing import Any

MAX_WAYPOINTS = 64
MAX_WAYPOINT_COORDINATE_M = 20.0
MAX_DISCRETE_ACTIONS = 64
TRAJECTORY_PREFIX_POINTS_TO_DROP = 3
FORWARD_STEP_M = 0.25
TURN_STEP_RAD = math.radians(15.0)
DEFAULT_VALID_FOR_S = 5.0


class InternVLAOutputError(ValueError):
    """The official agent returned an ambiguous or unsafe native result."""


@dataclass(frozen=True, slots=True)
class NativePrediction:
    """Neutral copy of the union returned by InternNav's real-world agent."""

    trajectory: object | None = None
    discrete_action: object | None = None
    pixel_goal: object | None = None


@dataclass(frozen=True, slots=True)
class InternVLAWaypoint:
    """One capture-time ``base_link`` waypoint, before public-contract mapping."""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class InternVLANavigationOutput:
    """Internal spatial result intentionally distinct from shared contracts."""

    observation_sequence: int
    waypoints: tuple[InternVLAWaypoint, ...]
    terminal: bool
    valid_for_s: float = DEFAULT_VALID_FOR_S
    confidence: float = 1.0
    frame: str = "base_link"

    def as_dict(self) -> dict[str, object]:
        """Return a transport-neutral mapping for provider integration."""

        return {
            "kind": "waypoint_plan",
            "observation_sequence": self.observation_sequence,
            "frame": self.frame,
            "waypoints": [
                {"x_m": point.x_m, "y_m": point.y_m, "yaw_rad": point.yaw_rad}
                for point in self.waypoints
            ],
            "confidence": self.confidence,
            "valid_for_s": self.valid_for_s,
            "terminal": self.terminal,
        }


def native_prediction_from_official(value: object) -> NativePrediction:
    """Copy the official object (or a test mapping) without retaining its type."""

    if isinstance(value, Mapping):
        return NativePrediction(
            trajectory=value.get("output_trajectory"),
            discrete_action=value.get("output_action"),
            pixel_goal=value.get("output_pixel"),
        )
    return NativePrediction(
        trajectory=getattr(value, "output_trajectory", None),
        discrete_action=getattr(value, "output_action", None),
        pixel_goal=getattr(value, "output_pixel", None),
    )


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
    if isinstance(value, bool):
        raise InternVLAOutputError(f"{name} must be a number")
    if not isinstance(value, Real):
        raise InternVLAOutputError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise InternVLAOutputError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise InternVLAOutputError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise InternVLAOutputError(f"{name} must be at most {maximum}")
    return result


def normalize_discrete_actions(value: object) -> list[int]:
    """Normalize Python or NumPy integer tokens without accepting floats."""

    actions = _as_list(value, "discrete_action")
    if not actions:
        raise InternVLAOutputError("discrete_action must not be empty")
    if len(actions) > MAX_DISCRETE_ACTIONS:
        raise InternVLAOutputError("discrete_action exceeds the supported horizon")
    normalized: list[int] = []
    for index, action in enumerate(actions):
        if isinstance(action, bool):
            raise InternVLAOutputError(f"discrete_action[{index}] must be an integer")
        try:
            normalized.append(operator.index(action))
        except TypeError as error:
            raise InternVLAOutputError(f"discrete_action[{index}] must be an integer") from error
    return normalized


def is_look_down_request(native: NativePrediction) -> bool:
    """Return whether the native result is exactly the internal LOOK_DOWN token."""

    return native.discrete_action is not None and normalize_discrete_actions(
        native.discrete_action
    ) == [5]


def _normalize_angle(value: float) -> float:
    normalized = (value + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(normalized, -math.pi) and value > 0.0:
        return math.pi
    return normalized


def _trajectory_waypoints(value: object) -> tuple[InternVLAWaypoint, ...]:
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
    waypoints: list[InternVLAWaypoint] = []
    for index, (x_m, y_m) in enumerate(coordinates):
        candidate = segment_headings[index] if index < len(segment_headings) else None
        if candidate is not None:
            latest_heading = candidate
        waypoints.append(InternVLAWaypoint(x_m=x_m, y_m=y_m, yaw_rad=latest_heading))
    return tuple(waypoints)


def _discrete_waypoints(actions: list[int]) -> tuple[InternVLAWaypoint, ...]:
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
    waypoints: list[InternVLAWaypoint] = []
    for action in actions:
        if action == 1:
            x_m += FORWARD_STEP_M * math.cos(yaw_rad)
            y_m += FORWARD_STEP_M * math.sin(yaw_rad)
        elif action == 2:
            yaw_rad = _normalize_angle(yaw_rad + TURN_STEP_RAD)
        else:
            yaw_rad = _normalize_angle(yaw_rad - TURN_STEP_RAD)
        waypoints.append(InternVLAWaypoint(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad))
    return tuple(waypoints)


def convert_native_prediction(
    native: NativePrediction,
    observation_sequence: int,
    *,
    valid_for_s: float = DEFAULT_VALID_FOR_S,
) -> InternVLANavigationOutput:
    """Convert exactly one native output arm while preserving spatial semantics."""

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
        return InternVLANavigationOutput(
            observation_sequence=observation_sequence,
            waypoints=_trajectory_waypoints(native.trajectory),
            terminal=False,
            valid_for_s=lifetime,
        )

    actions = normalize_discrete_actions(native.discrete_action)
    if actions == [0]:
        return InternVLANavigationOutput(
            observation_sequence=observation_sequence,
            waypoints=(),
            terminal=True,
            valid_for_s=lifetime,
        )
    return InternVLANavigationOutput(
        observation_sequence=observation_sequence,
        waypoints=_discrete_waypoints(actions),
        terminal=False,
        valid_for_s=lifetime,
    )


__all__ = [
    "DEFAULT_VALID_FOR_S",
    "FORWARD_STEP_M",
    "MAX_DISCRETE_ACTIONS",
    "MAX_WAYPOINTS",
    "TRAJECTORY_PREFIX_POINTS_TO_DROP",
    "TURN_STEP_RAD",
    "InternVLANavigationOutput",
    "InternVLAOutputError",
    "InternVLAWaypoint",
    "NativePrediction",
    "convert_native_prediction",
    "is_look_down_request",
    "native_prediction_from_official",
    "normalize_discrete_actions",
]
