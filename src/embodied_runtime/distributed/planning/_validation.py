"""Validation shared by planning state transitions."""

from __future__ import annotations

import math


def validate_staleness_check(
    current_observation_timestamp_s: float | None,
    max_observation_staleness_s: float | None,
) -> tuple[float | None, float | None]:
    if (current_observation_timestamp_s is None) != (max_observation_staleness_s is None):
        raise ValueError(
            "current_observation_timestamp_s and max_observation_staleness_s "
            "must be provided together"
        )
    if current_observation_timestamp_s is None:
        return None, None

    values: list[float] = []
    for name, value in (
        ("current_observation_timestamp_s", current_observation_timestamp_s),
        ("max_observation_staleness_s", max_observation_staleness_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        values.append(normalized)
    return values[0], values[1]
