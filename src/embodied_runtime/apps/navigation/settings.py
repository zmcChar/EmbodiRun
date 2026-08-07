"""Typed settings for the current Go2 navigation composition."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from embodied_runtime.robots.unitree.go2.profile import (
    MAX_LINEAR_SPEED_MPS,
    MAX_YAW_RATE_RPS,
)
from embodied_runtime.tasks.navigation import PlanarVelocityLimits

from .validation import nonempty, positive

BACKENDS = ("qwen", "streamvln", "internvla")
DEFAULT_GO2_CAMERA_URL = "http://127.0.0.1:8765"
DEFAULT_GO2_CONTROL_URL = "http://127.0.0.1:8080"
GO2_CONTROL_HARD_LIMITS = PlanarVelocityLimits(
    MAX_LINEAR_SPEED_MPS,
    MAX_LINEAR_SPEED_MPS,
    MAX_YAW_RATE_RPS,
)


@dataclass(frozen=True, slots=True)
class RunSettings:
    instruction: str | None = None
    episode_id: str = "go2-navigation"
    max_runtime_s: float = 120.0
    control_hz: float = 10.0
    lease_duration_s: float = 10.0
    max_events: int = 256
    session_mode: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instruction",
            nonempty(self.instruction, "instruction", optional=True),
        )
        object.__setattr__(self, "episode_id", nonempty(self.episode_id, "episode_id"))
        for name in ("max_runtime_s", "control_hz", "lease_duration_s"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        if isinstance(self.max_events, bool) or not isinstance(self.max_events, int):
            raise TypeError("max_events must be a positive integer")
        if self.max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if self.session_mode not in {"auto", "continuous", "reactive"}:
            raise ValueError("session_mode must be 'auto', 'continuous', or 'reactive'")


@dataclass(frozen=True, slots=True)
class PolicySettings:
    backend: str = "streamvln"
    qwen_base_url: str = "http://127.0.0.1:15003/v1"
    qwen_model: str = "qwen3.5-9b"
    qwen_api_key: str | None = None
    qwen_timeout_s: float = 60.0
    streamvln_root: str | None = None
    streamvln_model_path: str | None = None
    streamvln_device: str = "cuda:0"
    cuda_memory_fraction: float | None = 0.45
    max_new_tokens: int = 64
    local_files_only: bool = True
    warmup: bool = True
    internvla_root: str | None = None
    internvla_model_path: str | None = None
    internvla_variant: str = "dualvln"
    internvla_device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        for name in ("qwen_base_url", "qwen_model", "streamvln_device", "internvla_device"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        for name in (
            "qwen_api_key",
            "streamvln_root",
            "streamvln_model_path",
            "internvla_root",
            "internvla_model_path",
        ):
            object.__setattr__(self, name, nonempty(getattr(self, name), name, optional=True))
        object.__setattr__(self, "qwen_timeout_s", positive(self.qwen_timeout_s, "qwen_timeout_s"))
        if self.cuda_memory_fraction is not None:
            fraction = positive(self.cuda_memory_fraction, "cuda_memory_fraction")
            if fraction > 1.0:
                raise ValueError("cuda_memory_fraction must not exceed 1")
            object.__setattr__(self, "cuda_memory_fraction", fraction)
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be a positive integer")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.internvla_variant not in {"dualvln", "navdp"}:
            raise ValueError("internvla_variant must be 'dualvln' or 'navdp'")
        if not isinstance(self.local_files_only, bool) or not isinstance(self.warmup, bool):
            raise TypeError("local_files_only and warmup must be booleans")


@dataclass(frozen=True, slots=True)
class Go2Settings:
    camera_url: str = DEFAULT_GO2_CAMERA_URL
    control_url: str = DEFAULT_GO2_CONTROL_URL
    camera_token: str | None = None
    control_token: str | None = None
    camera_timeout_s: float = 2.0
    control_timeout_s: float = 1.0
    max_abs_vx_mps: float = 0.35
    max_abs_vy_mps: float = 0.35
    max_abs_yaw_rate_rps: float = 0.7
    position_tolerance_m: float = 0.08
    yaw_tolerance_rad: float = 0.08

    def __post_init__(self) -> None:
        for name in ("camera_url", "control_url"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
            parsed = urlsplit(getattr(self, name))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"{name} must be an absolute HTTP(S) URL")
        for name in ("camera_token", "control_token"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name, optional=True))
        for name in (
            "camera_timeout_s",
            "control_timeout_s",
            "max_abs_vx_mps",
            "max_abs_vy_mps",
            "max_abs_yaw_rate_rps",
            "position_tolerance_m",
            "yaw_tolerance_rad",
        ):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        requested = (self.max_abs_vx_mps, self.max_abs_vy_mps, self.max_abs_yaw_rate_rps)
        ceilings = (
            GO2_CONTROL_HARD_LIMITS.max_abs_vx_mps,
            GO2_CONTROL_HARD_LIMITS.max_abs_vy_mps,
            GO2_CONTROL_HARD_LIMITS.max_abs_yaw_rate_rps,
        )
        if any(value > ceiling for value, ceiling in zip(requested, ceilings, strict=True)):
            raise ValueError("Go2 limits cannot exceed vx=0.35, vy=0.35, yaw_rate=0.7")


@dataclass(frozen=True, slots=True)
class Go2NavigationAppConfig:
    run: RunSettings = RunSettings()
    policy: PolicySettings = PolicySettings()
    go2: Go2Settings = Go2Settings()


__all__ = [
    "BACKENDS",
    "DEFAULT_GO2_CAMERA_URL",
    "DEFAULT_GO2_CONTROL_URL",
    "GO2_CONTROL_HARD_LIMITS",
    "Go2NavigationAppConfig",
    "Go2Settings",
    "PolicySettings",
    "RunSettings",
]
