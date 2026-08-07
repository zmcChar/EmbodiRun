"""Validated TOML settings for the shared multi-robot cloud service."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from .._local_provider import LocalProviderConfig


@dataclass(frozen=True, slots=True)
class MultiRobotCloudConfig:
    provider: LocalProviderConfig = field(
        default_factory=lambda: LocalProviderConfig(
            provider_name="cloud-local",
            multi_tenant_safe=True,
            max_batch_size=1,
        )
    )
    host: str = "127.0.0.1"
    port: int = 18770
    request_timeout_s: float = 30.0
    max_sessions: int = 64
    max_pending_per_session: int = 1
    session_idle_ttl_s: float = 300.0
    action_space_id: str = "toy-vector-actions-v1"
    supported_embodiments: tuple[str, ...] = ("toy-vector",)
    shutdown_grace_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("cloud host must not be empty")
        if not 0 < self.port < 65536:
            raise ValueError("cloud port must be between 1 and 65535")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and greater than zero")
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")
        if self.max_pending_per_session <= 0:
            raise ValueError("max_pending_per_session must be greater than zero")
        if not math.isfinite(self.session_idle_ttl_s) or self.session_idle_ttl_s <= 0:
            raise ValueError("session_idle_ttl_s must be finite and greater than zero")
        if not self.action_space_id.strip():
            raise ValueError("action_space_id must not be empty")
        if not self.supported_embodiments:
            raise ValueError("supported_embodiments must not be empty")
        if not math.isfinite(self.shutdown_grace_s) or self.shutdown_grace_s < 0:
            raise ValueError("shutdown_grace_s must be finite and non-negative")


def load_multi_robot_cloud_config(path: str | Path) -> MultiRobotCloudConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    service = raw.get("service", {})
    provider = raw.get("provider", {})
    if not isinstance(service, Mapping) or not isinstance(provider, Mapping):
        raise TypeError("service and provider must be TOML tables")
    supported_embodiments = service.get("supported_embodiments", ("toy-vector",))
    if not isinstance(supported_embodiments, (list, tuple)):
        raise TypeError("service.supported_embodiments must be an array")
    return MultiRobotCloudConfig(
        provider=LocalProviderConfig.from_mapping(provider),
        host=str(service.get("host", "127.0.0.1")),
        port=int(service.get("port", 18770)),
        request_timeout_s=float(service.get("request_timeout_s", 30.0)),
        max_sessions=int(service.get("max_sessions", 64)),
        max_pending_per_session=int(service.get("max_pending_per_session", 1)),
        session_idle_ttl_s=float(service.get("session_idle_ttl_s", 300.0)),
        action_space_id=str(service.get("action_space_id", "toy-vector-actions-v1")),
        supported_embodiments=tuple(str(value) for value in supported_embodiments),
        shutdown_grace_s=float(service.get("shutdown_grace_s", 5.0)),
    )


__all__ = ["MultiRobotCloudConfig", "load_multi_robot_cloud_config"]
