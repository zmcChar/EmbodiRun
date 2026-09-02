"""Host-side configuration for a Unitree Go2 control adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .navigation.motion import PlanarVelocityLimits
from .profile import MAX_LINEAR_SPEED_MPS, MAX_YAW_RATE_RPS


def _default_velocity_limits() -> PlanarVelocityLimits:
    return PlanarVelocityLimits(
        max_abs_vx_mps=MAX_LINEAR_SPEED_MPS,
        max_abs_vy_mps=MAX_LINEAR_SPEED_MPS,
        max_abs_yaw_rate_rps=MAX_YAW_RATE_RPS,
    )


@dataclass(frozen=True, slots=True)
class Go2Config:
    """Connection and motion bounds for one Go2 control agent."""

    control_url: str
    robot_id: str = "go2"
    api_token: str | None = field(default=None, repr=False)
    timeout_s: float = 1.0
    expected_transport: str | None = "unitree-sdk2"
    velocity_lease_duration_s: float = 10.0
    velocity_limits: PlanarVelocityLimits = field(
        default_factory=_default_velocity_limits
    )

    def __post_init__(self) -> None:
        parsed_url = urlsplit(self.control_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("control_url must be an absolute HTTP(S) URL")
        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        for name in ("timeout_s", "velocity_lease_duration_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.expected_transport is not None and not self.expected_transport.strip():
            raise ValueError("expected_transport must not be empty when provided")
        if not isinstance(self.velocity_limits, PlanarVelocityLimits):
            raise TypeError("velocity_limits must be PlanarVelocityLimits")


__all__ = ["Go2Config"]
