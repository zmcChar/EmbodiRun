from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.apps.cloud_edge.pi05_collaboration import (
    run_pi05_cpu_gpu_collaboration,
)
from embodied_runtime.apps.cloud_edge.pi05_settings import (
    Pi05CpuGpuConfig,
)
from embodied_runtime.distributed import FailoverConfig, FailoverMode


@pytest.mark.pi05
def test_real_pi05_gpu_and_cpu_models_blend_then_fall_back() -> None:
    checkpoint = os.environ.get("EMBODIED_RUNTIME_PI05_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set EMBODIED_RUNTIME_PI05_CHECKPOINT to a local π0.5 checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("real CPU/GPU collaboration requires CUDA")

    summary = run_pi05_cpu_gpu_collaboration(
        Pi05CpuGpuConfig(
            checkpoint=checkpoint,
            num_steps=1,
            language_length=8,
            cloud_weight=0.5,
            failover=FailoverConfig(
                mode=FailoverMode.ASYNC_BLEND,
                cloud_request_timeout_s=5.0,
                cloud_result_ttl_s=5.0,
                max_cloud_sequence_lag=4,
                cloud_submit_interval_s=60.0,
            ),
        )
    )

    assert summary["cloud_device"] == "cuda:0"
    assert summary["edge_device"] == "cpu"
    assert summary["first_tick_source"] == "edge"
    assert summary["cloud_in_flight_after_first"]
    assert summary["second_tick_source"] == "blended"
    assert summary["disconnected_tick_source"] == "edge"
    assert summary["disconnected_reason"] == "cloud_disconnected"
    assert summary["action_shape"] == (50, 32)
    assert summary["final_action_device"] == "cpu"
    assert summary["fusion_max_abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert summary["fused_vs_edge_max_abs"] > 0.0
    assert summary["fused_vs_cloud_max_abs"] > 0.0
