from __future__ import annotations

from embodied_runtime.apps.cloud_edge_failover import load_demo_config
from embodied_runtime.apps.pi05_cpu_gpu_collaboration import (
    load_pi05_cpu_gpu_config,
)
from embodied_runtime.distributed import FailoverMode


def test_toy_config_loads_async_blend_and_weight(tmp_path) -> None:
    config_path = tmp_path / "blend.toml"
    config_path.write_text(
        """
[failover]
mode = "async_blend"

[fusion]
cloud_weight = 0.25
""",
        encoding="utf-8",
    )

    config = load_demo_config(config_path)

    assert config.failover.mode is FailoverMode.ASYNC_BLEND
    assert config.cloud_weight == 0.25


def test_real_cpu_gpu_config_declares_physical_placements() -> None:
    config = load_pi05_cpu_gpu_config("configs/pi05_cpu_gpu_collaboration.toml")

    assert config.failover.mode is FailoverMode.ASYNC_BLEND
    assert config.cloud_device == "cuda:0"
    assert config.edge_device == "cpu"
    assert config.cloud_weight == 0.5
