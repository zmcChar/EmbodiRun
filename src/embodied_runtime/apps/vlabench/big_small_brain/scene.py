"""Simulator state inspection for the get-coffee task."""

from __future__ import annotations

from typing import Any


def coffee_mug_is_placed(endpoint: Any) -> bool:
    environment = endpoint.raw_environment
    inner = getattr(environment, "_env", None)
    task = getattr(inner, "task", None)
    physics = getattr(inner, "physics", None)
    conditions = getattr(getattr(task, "conditions", None), "conditions", None)
    if not conditions:
        raise RuntimeError("get_coffee does not expose its contain condition")
    is_met = getattr(conditions[0], "is_met", None)
    if not callable(is_met):
        raise TypeError("get_coffee contain condition does not expose is_met()")
    return bool(is_met(physics))
