"""Validated TOML settings for one robot-scoped edge runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from embodied_runtime.distributed import FailoverConfig, FailoverMode
from embodied_runtime.distributed.session import RobotSessionIdentity

from .._local_provider import LocalProviderConfig

SUPPORTED_EDGE_MODES = frozenset((FailoverMode.ASYNC_CLOUD_PREFERRED, FailoverMode.EDGE_ONLY))
OBSERVATION_MODES = frozenset(("target_vector", "adapter_synthetic"))


@dataclass(frozen=True, slots=True)
class MultiRobotEdgeConfig:
    identity: RobotSessionIdentity
    provider: LocalProviderConfig = field(
        default_factory=lambda: LocalProviderConfig(provider_name="edge-local")
    )
    cloud_host: str = "localhost"
    cloud_port: int = 18770
    cloud_timeout_s: float = 5.0
    registration_timeout_s: float = 2.0
    reconnect_interval_s: float = 1.0
    physical_host_id: str = "edge-host"
    physical_resource_id: str = "local-device-0"
    failover: FailoverConfig = field(
        default_factory=lambda: FailoverConfig(
            mode=FailoverMode.ASYNC_CLOUD_PREFERRED,
            cloud_request_timeout_s=5.0,
            cloud_result_ttl_s=2.0,
            max_cloud_sequence_lag=2,
            cloud_submit_interval_s=0.0,
        )
    )
    ticks: int = 8
    control_period_s: float = 0.05
    observation_mode: str = "target_vector"
    observation_offset: float = 0.0
    observation_seed: int = 0
    observation_language_length: int = 48
    disconnect_tick: int | None = None
    reconnect_tick: int | None = None

    def __post_init__(self) -> None:
        if not self.cloud_host:
            raise ValueError("cloud_host must not be empty")
        if not 0 < self.cloud_port < 65536:
            raise ValueError("cloud_port must be between 1 and 65535")
        if not math.isfinite(self.cloud_timeout_s) or self.cloud_timeout_s <= 0:
            raise ValueError("cloud_timeout_s must be finite and greater than zero")
        if not math.isfinite(self.registration_timeout_s) or self.registration_timeout_s <= 0:
            raise ValueError("registration_timeout_s must be finite and greater than zero")
        if not math.isfinite(self.reconnect_interval_s) or self.reconnect_interval_s < 0:
            raise ValueError("reconnect_interval_s must be finite and non-negative")
        if not self.physical_host_id.strip():
            raise ValueError("physical_host_id must not be empty")
        if not self.physical_resource_id.strip():
            raise ValueError("physical_resource_id must not be empty")
        if self.failover.mode not in SUPPORTED_EDGE_MODES:
            raise ValueError(
                "multi-robot edge supports async_cloud_preferred or edge_only; "
                "numeric blending requires an explicit shared action contract"
            )
        if self.ticks <= 0:
            raise ValueError("ticks must be greater than zero")
        if not math.isfinite(self.control_period_s) or self.control_period_s < 0:
            raise ValueError("control_period_s must be finite and non-negative")
        if self.observation_mode not in OBSERVATION_MODES:
            supported = ", ".join(sorted(OBSERVATION_MODES))
            raise ValueError(f"observation_mode must be one of: {supported}")
        if not math.isfinite(self.observation_offset):
            raise ValueError("observation_offset must be finite")
        if not isinstance(self.observation_seed, int) or isinstance(self.observation_seed, bool):
            raise TypeError("observation_seed must be an integer")
        if (
            not isinstance(self.observation_language_length, int)
            or isinstance(self.observation_language_length, bool)
            or self.observation_language_length <= 0
        ):
            raise ValueError("observation_language_length must be a positive integer")


def load_multi_robot_edge_config(path: str | Path) -> MultiRobotEdgeConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    identity = raw.get("identity", {})
    provider = raw.get("edge_provider", {})
    cloud = raw.get("cloud", {})
    coordination = raw.get("coordination", {})
    demo = raw.get("demo", {})
    observation = raw.get("observation", {})
    for name, values in (
        ("identity", identity),
        ("edge_provider", provider),
        ("cloud", cloud),
        ("coordination", coordination),
        ("demo", demo),
        ("observation", observation),
    ):
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a TOML table")

    cloud_timeout_s = float(cloud.get("timeout_s", 5.0))
    return MultiRobotEdgeConfig(
        identity=RobotSessionIdentity(
            robot_id=str(identity.get("robot_id", "robot-demo")),
            edge_node_id=str(identity.get("edge_node_id", "edge-demo")),
            session_id=str(identity.get("session_id", "session-demo")),
            embodiment=str(identity.get("embodiment", "toy-vector")),
            action_space_id=str(identity.get("action_space_id", "toy-vector-actions-v1")),
        ),
        provider=LocalProviderConfig.from_mapping(provider),
        cloud_host=str(cloud.get("host", "localhost")),
        cloud_port=int(cloud.get("port", 18770)),
        cloud_timeout_s=cloud_timeout_s,
        registration_timeout_s=float(cloud.get("registration_timeout_s", 2.0)),
        reconnect_interval_s=float(cloud.get("reconnect_interval_s", 1.0)),
        physical_host_id=str(identity.get("physical_host_id", "edge-host")),
        physical_resource_id=str(identity.get("physical_resource_id", "local-device-0")),
        failover=FailoverConfig(
            mode=FailoverMode(coordination.get("mode", FailoverMode.ASYNC_CLOUD_PREFERRED.value)),
            cloud_request_timeout_s=float(
                coordination.get("cloud_request_timeout_s", cloud_timeout_s)
            ),
            cloud_result_ttl_s=float(coordination.get("cloud_result_ttl_s", 2.0)),
            max_cloud_sequence_lag=int(coordination.get("max_cloud_sequence_lag", 2)),
            cloud_submit_interval_s=float(coordination.get("cloud_submit_interval_s", 0.0)),
        ),
        ticks=int(demo.get("ticks", 8)),
        control_period_s=float(demo.get("control_period_s", 0.05)),
        observation_mode=str(observation.get("mode", "target_vector")),
        observation_offset=float(observation.get("offset", 0.0)),
        observation_seed=int(observation.get("seed", 0)),
        observation_language_length=int(observation.get("language_length", 48)),
        disconnect_tick=_optional_int(demo.get("disconnect_tick")),
        reconnect_tick=_optional_int(demo.get("reconnect_tick")),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "OBSERVATION_MODES",
    "MultiRobotEdgeConfig",
    "load_multi_robot_edge_config",
]
