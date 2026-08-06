"""StreamVLN's discrete Go2 action vocabulary and spatial conversion.

The real-world StreamVLN checkpoint emits four integer actions: stop, move
forward 25 centimetres, and turn left or right by 15 degrees.  This module is
dependency-free so action validation and Go2 path conversion remain usable in
lightweight serving and protocol processes.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

FORWARD_STEP_M = 0.25
TURN_STEP_RAD = math.radians(15.0)
MAX_FUTURE_ACTIONS = 4


class StreamVLNAction(IntEnum):
    """Native action ids used by the official real-world checkpoint."""

    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class StreamVLNNativeOutputError(ValueError):
    """The checkpoint returned an invalid native action sequence."""


@dataclass(frozen=True, slots=True)
class StreamVLNWaypoint:
    """One cumulative target pose in the image capture-time ``base_link``."""

    x_m: float
    y_m: float
    yaw_rad: float

    def as_dict(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m, "yaw_rad": self.yaw_rad}


@dataclass(frozen=True, slots=True)
class StreamVLNWaypointPlan:
    """Model-local path representation before conversion to public contracts."""

    waypoints: tuple[StreamVLNWaypoint, ...]
    terminal: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "waypoints": [waypoint.as_dict() for waypoint in self.waypoints],
            "terminal": self.terminal,
        }


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def normalize_native_actions(
    actions: Sequence[object],
    *,
    max_actions: int = MAX_FUTURE_ACTIONS,
) -> tuple[int, ...]:
    """Validate an evaluator result without accepting lossy numeric coercion."""

    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise StreamVLNNativeOutputError("StreamVLN actions must be a sequence")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    if not actions:
        raise StreamVLNNativeOutputError("StreamVLN returned no parseable actions")

    normalized: list[int] = []
    for index, action in enumerate(actions):
        if isinstance(action, bool):
            raise StreamVLNNativeOutputError(f"StreamVLN action {index} must be an integer")
        try:
            action_id = operator.index(action)
        except TypeError as error:
            raise StreamVLNNativeOutputError(
                f"StreamVLN action {index} must be an integer"
            ) from error
        try:
            StreamVLNAction(action_id)
        except ValueError as error:
            raise StreamVLNNativeOutputError(
                f"unsupported StreamVLN action {action_id} at index {index}"
            ) from error
        normalized.append(action_id)
        if len(normalized) > max_actions:
            raise StreamVLNNativeOutputError(
                f"StreamVLN returned more than {max_actions} reachable actions"
            )
        if action_id == StreamVLNAction.STOP:
            # The official controller ignores any suffix after STOP.  Truncate
            # here as well so downstream consumers never see unreachable work.
            break
    return tuple(normalized)


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def actions_to_cumulative_waypoints(
    actions: Sequence[object],
    *,
    forward_step_m: float = FORWARD_STEP_M,
    turn_step_rad: float = TURN_STEP_RAD,
    max_actions: int = MAX_FUTURE_ACTIONS,
) -> StreamVLNWaypointPlan:
    """Convert native actions into cumulative capture-time Go2 target poses.

    Turns intentionally produce waypoints: the Go2 path tracker must complete
    the requested heading before it consumes the following forward target.  A
    bare STOP is therefore the only valid empty plan.
    """

    forward_step = _positive_finite(forward_step_m, "forward_step_m")
    turn_step = _positive_finite(turn_step_rad, "turn_step_rad")
    normalized = normalize_native_actions(actions, max_actions=max_actions)

    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0
    terminal = False
    waypoints: list[StreamVLNWaypoint] = []
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
        waypoints.append(StreamVLNWaypoint(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad))

    return StreamVLNWaypointPlan(waypoints=tuple(waypoints), terminal=terminal)


__all__ = [
    "FORWARD_STEP_M",
    "MAX_FUTURE_ACTIONS",
    "TURN_STEP_RAD",
    "StreamVLNAction",
    "StreamVLNNativeOutputError",
    "StreamVLNWaypoint",
    "StreamVLNWaypointPlan",
    "actions_to_cumulative_waypoints",
    "normalize_native_actions",
]
