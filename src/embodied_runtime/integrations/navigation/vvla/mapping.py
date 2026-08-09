"""Translate ActiveVLN's discrete R2R actions into task-owned waypoints."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from embodied_runtime.tasks.navigation import Waypoint, WaypointPlan


class ActiveVLNAction(Protocol):
    """Structural subset of ``vvla``'s ``NavigationAction``."""

    name: str
    value: int | None


@dataclass(frozen=True, slots=True)
class CanonicalActiveVLNAction:
    """Backend-neutral action produced by the strict deployment parser."""

    name: str
    value: int | None


@dataclass(frozen=True, slots=True)
class ActiveVLNPrediction:
    """Backend-neutral prediction record shared by local ActiveVLN runtimes."""

    actions: tuple[ActiveVLNAction, ...]
    text: str
    latency_ms: float
    token_ids: tuple[int, ...]


_ALLOWED_VALUES: dict[str, frozenset[int | None]] = {
    "move forward": frozenset({25, 50, 75}),
    "turn left": frozenset({15, 30, 45}),
    "turn right": frozenset({15, 30, 45}),
    "stop": frozenset({None}),
}
MAX_ACTIVEVLN_FORWARD_M = 0.75
MAX_ACTIVEVLN_TURN_RAD = math.radians(45.0)
_CANONICAL_ACTION = re.compile(
    r"(?:move forward (25|50|75)cm|turn (left|right) (15|30|45) degrees|stop)"
)


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def parse_canonical_activevln_actions(text: str) -> tuple[CanonicalActiveVLNAction, ...]:
    """Parse only the exact pinned R2R response grammar."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("ActiveVLN returned empty action text")
    fragments = [fragment.strip() for fragment in text.strip().split(",")]
    if any(not fragment for fragment in fragments) or len(fragments) > 3:
        raise ValueError(f"ActiveVLN returned non-canonical action text: {text!r}")

    parsed: list[CanonicalActiveVLNAction] = []
    for fragment in fragments:
        match = _CANONICAL_ACTION.fullmatch(fragment)
        if match is None:
            raise ValueError(f"ActiveVLN returned non-canonical action text: {text!r}")
        forward, direction, degrees = match.groups()
        if forward is not None:
            parsed.append(CanonicalActiveVLNAction("move forward", int(forward)))
        elif direction is not None:
            parsed.append(CanonicalActiveVLNAction(f"turn {direction}", int(degrees)))
        else:
            parsed.append(CanonicalActiveVLNAction("stop", None))

    if any(action.name == "stop" for action in parsed[:-1]):
        raise ValueError("ActiveVLN stop action must be the final action")
    return tuple(parsed)


def validate_activevln_action_text(text: str, actions: Sequence[ActiveVLNAction]) -> None:
    """Require canonical checkpoint output instead of the upstream fuzzy parser.

    VVLA's parser deliberately accepts missing magnitudes and fragments that only
    contain an action keyword.  That is useful for offline evaluation, but it is
    too permissive at a robot actuation boundary.  The deployment adapter accepts
    only the exact R2R prompt grammar and verifies that the typed parse agrees.
    """

    expected = parse_canonical_activevln_actions(text)
    actual = [(action.name, action.value) for action in actions]
    if actual != [(action.name, action.value) for action in expected]:
        raise ValueError("ActiveVLN parsed actions do not match its canonical action text")


def activevln_actions_to_waypoint_plan(
    actions: Sequence[ActiveVLNAction],
    *,
    observation_sequence: int,
    valid_for_s: float = 5.0,
) -> WaypointPlan:
    """Convert one native action chunk into cumulative capture-frame waypoints.

    ActiveVLN's magnitudes are centimetres for forward commands and degrees for
    turns.  Only the checkpoint's pinned R2R grammar is accepted; an unexpected
    magnitude fails closed instead of becoming an unsafe robot command.
    """

    if not actions:
        raise ValueError("ActiveVLN returned no navigation actions")
    if len(actions) > 3:
        raise ValueError("ActiveVLN returned more than three navigation actions")

    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0
    terminal = False
    waypoints: list[Waypoint] = []
    for index, action in enumerate(actions):
        name = action.name
        value = action.value
        allowed = _ALLOWED_VALUES.get(name)
        value_has_expected_type = value is None if name == "stop" else type(value) is int
        if allowed is None or not value_has_expected_type or value not in allowed:
            raise ValueError(f"unsupported ActiveVLN action at index {index}: {name!r} {value!r}")
        if name == "stop":
            if index != len(actions) - 1:
                raise ValueError("ActiveVLN stop action must be the final action")
            terminal = True
            break
        if name == "move forward":
            distance_m = float(value) / 100.0
            x_m += distance_m * math.cos(yaw_rad)
            y_m += distance_m * math.sin(yaw_rad)
        elif name == "turn left":
            yaw_rad = _normalize_angle(yaw_rad + math.radians(float(value)))
        else:
            yaw_rad = _normalize_angle(yaw_rad - math.radians(float(value)))
        waypoints.append(Waypoint(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad))

    return WaypointPlan(
        observation_sequence=observation_sequence,
        waypoints=tuple(waypoints),
        terminal=terminal,
        confidence=1.0,
        valid_for_s=valid_for_s,
    )


__all__ = [
    "MAX_ACTIVEVLN_FORWARD_M",
    "MAX_ACTIVEVLN_TURN_RAD",
    "ActiveVLNAction",
    "ActiveVLNPrediction",
    "CanonicalActiveVLNAction",
    "activevln_actions_to_waypoint_plan",
    "parse_canonical_activevln_actions",
    "validate_activevln_action_text",
]
