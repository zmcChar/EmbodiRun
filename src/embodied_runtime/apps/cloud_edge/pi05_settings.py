"""Configuration parsing for the pi0.5 CPU/GPU collaboration app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from embodied_runtime.distributed import FailoverConfig, FailoverMode


@dataclass(frozen=True, slots=True)
class Pi05CpuGpuConfig:
    checkpoint: str = "lerobot/pi05_base"
    cloud_device: str = "cuda:0"
    edge_device: str = "cpu"
    dtype: str = "preserve"
    num_steps: int = 10
    language_length: int = 8
    seed: int = 0
    cuda_graph: bool = False
    local_files_only: bool = True
    one_way_latency_s: float = 0.0
    cloud_weight: float = 0.5
    failover: FailoverConfig = field(
        default_factory=lambda: FailoverConfig(
            mode=FailoverMode.ASYNC_BLEND,
            cloud_request_timeout_s=5.0,
            cloud_result_ttl_s=5.0,
            max_cloud_sequence_lag=4,
            cloud_submit_interval_s=60.0,
        )
    )

    def __post_init__(self) -> None:
        if not self.cloud_device.startswith("cuda:"):
            raise ValueError("cloud_device must be cuda:N for the GPU collaboration demo")
        if self.edge_device != "cpu":
            raise ValueError("edge_device must be cpu for the CPU collaboration demo")
        if self.dtype not in {"preserve", "float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        if self.language_length <= 0:
            raise ValueError("language_length must be greater than zero")
        if self.one_way_latency_s < 0:
            raise ValueError("one_way_latency_s cannot be negative")
        if not 0.0 <= self.cloud_weight <= 1.0:
            raise ValueError("cloud_weight must be between zero and one")


def load_pi05_cpu_gpu_config(path: str | Path) -> Pi05CpuGpuConfig:
    """Load model placement, coordination, and dummy-link settings from TOML."""

    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    model = raw.get("model", {})
    cloud = raw.get("cloud", {})
    edge = raw.get("edge", {})
    coordination = raw.get("coordination", {})
    link = raw.get("dummy_link", {})
    return Pi05CpuGpuConfig(
        checkpoint=str(model.get("checkpoint", "lerobot/pi05_base")),
        cloud_device=str(cloud.get("device", "cuda:0")),
        edge_device=str(edge.get("device", "cpu")),
        dtype=str(cloud.get("dtype", "preserve")),
        num_steps=int(cloud.get("num_steps", 10)),
        language_length=int(cloud.get("language_length", 8)),
        seed=int(cloud.get("seed", 0)),
        cuda_graph=bool(cloud.get("cuda_graph", False)),
        local_files_only=bool(model.get("local_files_only", True)),
        one_way_latency_s=float(link.get("one_way_latency_s", 0.0)),
        cloud_weight=float(coordination.get("cloud_weight", 0.5)),
        failover=FailoverConfig(
            mode=FailoverMode(coordination.get("mode", FailoverMode.ASYNC_BLEND)),
            cloud_request_timeout_s=float(coordination.get("cloud_request_timeout_s", 5.0)),
            cloud_result_ttl_s=float(coordination.get("cloud_result_ttl_s", 5.0)),
            max_cloud_sequence_lag=int(coordination.get("max_cloud_sequence_lag", 4)),
            cloud_submit_interval_s=float(coordination.get("cloud_submit_interval_s", 60.0)),
        ),
    )
