"""InternVLA official-output copies and native discrete-action validation."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_DISCRETE_ACTIONS = 64


class InternVLAOutputError(ValueError):
    """The official agent returned an ambiguous or unsafe native result."""


@dataclass(frozen=True, slots=True)
class NativePrediction:
    """Neutral copy of the union returned by InternNav's real-world agent."""

    trajectory: object | None = None
    discrete_action: object | None = None
    pixel_goal: object | None = None


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


__all__ = [
    "MAX_DISCRETE_ACTIONS",
    "InternVLAOutputError",
    "NativePrediction",
    "is_look_down_request",
    "native_prediction_from_official",
    "normalize_discrete_actions",
]
