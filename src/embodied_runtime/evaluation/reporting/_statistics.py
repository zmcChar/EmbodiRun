"""Small dependency-free statistics used by evaluation summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

_WILSON_Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    proportion = successes / trials
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        _WILSON_Z_95
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile over a non-empty sequence."""

    if not values:
        raise ValueError("a percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def percentile_or_none(
    values: Sequence[float],
    percentile_value: float,
) -> float | None:
    return percentile(values, percentile_value) if values else None


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
