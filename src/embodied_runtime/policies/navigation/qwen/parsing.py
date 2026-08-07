"""Strict parsing of Qwen waypoint-plan responses."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from embodied_runtime.tasks.navigation import (
    MAX_WAYPOINTS,
    NavigationContractError,
    Waypoint,
    WaypointPlan,
)

from .errors import QwenNavigationValidationError
from .schema import PLAN_FIELDS, WAYPOINT_FIELDS


def decode_strict_json_object(content: str) -> dict[str, Any]:
    """Decode one assistant response without accepting ambiguous JSON."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise QwenNavigationValidationError(
            f"Qwen assistant content is not strict JSON: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise QwenNavigationValidationError("Qwen assistant content must be one JSON object")
    return decoded


def _exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = expected.difference(payload)
    unknown = set(payload).difference(expected)
    if missing or unknown:
        raise QwenNavigationValidationError(
            f"{name} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_waypoint_plan(
    payload: Mapping[str, Any],
    *,
    observation_sequence: int,
) -> WaypointPlan:
    """Validate one decoded model object and build the shared result contract."""

    if not isinstance(payload, Mapping):
        raise QwenNavigationValidationError("Qwen navigation response must be an object")
    _exact_fields(payload, PLAN_FIELDS, name="waypoint plan")

    frame = payload["frame"]
    if frame != "base_link":
        raise QwenNavigationValidationError("waypoint plan frame must be 'base_link'")

    raw_waypoints = payload["waypoints"]
    if isinstance(raw_waypoints, (str, bytes)) or not isinstance(raw_waypoints, Sequence):
        raise QwenNavigationValidationError("waypoints must be an array")
    if len(raw_waypoints) > MAX_WAYPOINTS:
        raise QwenNavigationValidationError(f"waypoints may contain at most {MAX_WAYPOINTS} items")

    waypoints: list[Waypoint] = []
    for index, raw_waypoint in enumerate(raw_waypoints):
        if not isinstance(raw_waypoint, Mapping):
            raise QwenNavigationValidationError(f"waypoints[{index}] must be an object")
        _exact_fields(raw_waypoint, WAYPOINT_FIELDS, name=f"waypoints[{index}]")
        try:
            waypoints.append(
                Waypoint(
                    x_m=raw_waypoint["x_m"],
                    y_m=raw_waypoint["y_m"],
                    yaw_rad=raw_waypoint["yaw_rad"],
                )
            )
        except (NavigationContractError, TypeError, ValueError) as error:
            raise QwenNavigationValidationError(
                f"waypoints[{index}] is invalid: {error}"
            ) from error

    terminal = payload["terminal"]
    if type(terminal) is not bool:
        raise QwenNavigationValidationError("terminal must be a boolean")

    try:
        return WaypointPlan(
            observation_sequence=observation_sequence,
            waypoints=tuple(waypoints),
            terminal=terminal,
            confidence=payload["confidence"],
            valid_for_s=payload["valid_for_s"],
            frame=frame,
        )
    except (NavigationContractError, TypeError, ValueError) as error:
        raise QwenNavigationValidationError(f"invalid waypoint plan: {error}") from error


__all__ = ["decode_strict_json_object", "validate_waypoint_plan"]
