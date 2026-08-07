"""Settings for asynchronous SmolVLA edge and pi0.5 cloud inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from embodied_runtime.distributed import FailoverConfig, FailoverMode

SUPPORTED_MODES = frozenset((FailoverMode.ASYNC_CLOUD_PREFERRED, FailoverMode.EDGE_ONLY))


@dataclass(frozen=True, slots=True)
class SmolVLAPi05AsyncConfig:
    """Placement and coordination settings for the real two-host prototype."""

    edge_checkpoint: str
    edge_vlm_base_path: str
    cloud_host: str
    cloud_port: int = 18765
    edge_device: str = "cuda:0"
    stats_variant: str = "so100"
    num_steps: int = 10
    language_length: int = 48
    seed: int = 0
    local_files_only: bool = True
    tcp_timeout_s: float = 120.0
    failover: FailoverConfig = field(
        default_factory=lambda: FailoverConfig(
            mode=FailoverMode.ASYNC_CLOUD_PREFERRED,
            cloud_request_timeout_s=120.0,
            cloud_result_ttl_s=120.0,
            max_cloud_sequence_lag=4,
            # Keep the completed result cached for the second demo tick.
            cloud_submit_interval_s=60.0,
        )
    )

    def __post_init__(self) -> None:
        if not self.edge_checkpoint:
            raise ValueError("edge_checkpoint must not be empty")
        if not self.edge_vlm_base_path:
            raise ValueError("edge_vlm_base_path must not be empty")
        if not self.cloud_host:
            raise ValueError("cloud_host must not be empty")
        if not 0 < self.cloud_port < 65536:
            raise ValueError("cloud_port must be between 1 and 65535")
        if not self.edge_device.startswith("cuda:"):
            raise ValueError("edge_device must be cuda:N")
        if not self.stats_variant:
            raise ValueError("stats_variant must not be empty")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        if self.language_length <= 0:
            raise ValueError("language_length must be greater than zero")
        if not math.isfinite(self.tcp_timeout_s) or self.tcp_timeout_s <= 0:
            raise ValueError("tcp_timeout_s must be greater than zero")
        if self.failover.mode not in SUPPORTED_MODES:
            raise ValueError(
                "SmolVLA/pi0.5 supports only async_cloud_preferred or edge_only; "
                "async_blend is invalid because the action semantics differ"
            )
