"""Typed Unitree Go2 observation and base-command values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from embodied_runtime.contracts import Metadata
from embodied_runtime.utils import Pose2D


def _bounded(value: object, name: str, limit: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > limit:
        raise ValueError(f"{name} must be finite and within +/-{limit}")
    return result


@dataclass(frozen=True, slots=True)
class Go2Limits:
    # Keep these defaults aligned with go2-control-api's hard admission
    # bounds.  Commands above them are rejected instead of clipped by the
    # robot service, which would otherwise break the controller heartbeat.
    max_abs_vx_mps: float = 0.35
    max_abs_vy_mps: float = 0.35
    max_abs_yaw_rate_rps: float = 0.7

    def __post_init__(self) -> None:
        for name in ("max_abs_vx_mps", "max_abs_vy_mps", "max_abs_yaw_rate_rps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))


DEFAULT_GO2_LIMITS = Go2Limits()


@dataclass(frozen=True, slots=True)
class BaseVelocityCommand:
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float
    limits: Go2Limits = DEFAULT_GO2_LIMITS

    def __post_init__(self) -> None:
        if not isinstance(self.limits, Go2Limits):
            raise TypeError("limits must be Go2Limits")
        object.__setattr__(
            self, "vx_mps", _bounded(self.vx_mps, "vx_mps", self.limits.max_abs_vx_mps)
        )
        object.__setattr__(
            self, "vy_mps", _bounded(self.vy_mps, "vy_mps", self.limits.max_abs_vy_mps)
        )
        object.__setattr__(
            self,
            "yaw_rate_rps",
            _bounded(
                self.yaw_rate_rps,
                "yaw_rate_rps",
                self.limits.max_abs_yaw_rate_rps,
            ),
        )

    @classmethod
    def stopped(cls, limits: Go2Limits = DEFAULT_GO2_LIMITS) -> BaseVelocityCommand:
        return cls(0.0, 0.0, 0.0, limits)

    @property
    def moving(self) -> bool:
        return any(abs(value) > 1e-6 for value in (self.vx_mps, self.vy_mps, self.yaw_rate_rps))


@dataclass(frozen=True, slots=True)
class Go2State:
    pose: Pose2D
    forward_velocity_mps: float
    lateral_velocity_mps: float
    yaw_rate_rps: float
    sequence: int
    received_at_s: float | None
    active_action: Mapping[str, object] | None = None
    raw: Metadata = field(default_factory=dict)


__all__ = [
    "DEFAULT_GO2_LIMITS",
    "BaseVelocityCommand",
    "Go2Limits",
    "Go2State",
]
