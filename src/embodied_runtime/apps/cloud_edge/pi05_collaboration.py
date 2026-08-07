"""Async coordinator and evidence reporting for CPU/GPU collaboration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverMode,
    ResultSource,
)
from embodied_runtime.distributed.communication import (
    DummyLink,
    DummyRemoteInferenceEndpoint,
)
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult
from embodied_runtime.models.request import RawRequest

from ..action_fusion import make_action_result_fuser
from .pi05_resources import (
    Pi05CollaborationResources,
    build_pi05_collaboration_resources,
)
from .pi05_settings import Pi05CpuGpuConfig


def run_pi05_cpu_gpu_collaboration(config: Pi05CpuGpuConfig) -> dict[str, Any]:
    """Build both placed models and run edge, blended, and disconnected ticks."""

    resources = build_pi05_collaboration_resources(config)
    try:
        return asyncio.run(run_pi05_collaboration(resources, config))
    except Exception:
        resources.close()
        raise


async def run_pi05_collaboration(
    resources: Pi05CollaborationResources,
    config: Pi05CpuGpuConfig,
) -> dict[str, Any]:
    torch = _torch()
    link = DummyLink(one_way_latency_s=config.one_way_latency_s)
    remote_cloud = DummyRemoteInferenceEndpoint(resources.cloud_engine, link)
    base_fuser = make_action_result_fuser(cloud_weight=config.cloud_weight)
    fusion_evidence: dict[str, Any] = {}

    def record_fusion(
        edge_result: InferenceResult,
        cloud_result: InferenceResult,
    ) -> InferenceResult:
        fused = base_fuser(edge_result, cloud_result)
        fusion_evidence["edge"] = actions(edge_result).detach().float().cpu().clone()
        fusion_evidence["cloud"] = actions(cloud_result).detach().float().cpu().clone()
        fusion_evidence["fused"] = actions(fused).detach().float().cpu().clone()
        return fused

    coordinator = AsyncFailoverCoordinator(
        edge=resources.edge_engine,
        cloud=remote_cloud,
        config=config.failover,
        fuser=record_fusion,
    )
    cloud_payload = resources.cloud_adapter.synthetic_batch(
        batch_size=1,
        language_length=config.language_length,
        seed=config.seed,
    )

    try:
        await resources.edge_engine.start()
        await resources.cloud_engine.start()
        torch.cuda.reset_peak_memory_stats(torch.device(config.cloud_device))
        first_started = time.perf_counter()
        first = await coordinator.infer_async(
            edge_request(resources.edge_adapter, tick=1),
            cloud_request=cloud_request(
                cloud_payload,
                tick=1,
                num_steps=config.num_steps,
                seed=config.seed,
            ),
        )
        first_tick_time_s = time.perf_counter() - first_started
        cloud_in_flight_after_first = coordinator.cloud_request_in_flight

        if config.failover.mode is not FailoverMode.EDGE_ONLY:
            await asyncio.wait_for(
                coordinator.wait_for_cloud_idle(),
                timeout=config.failover.cloud_request_timeout_s + 5.0,
            )

        second = await coordinator.infer_async(
            edge_request(resources.edge_adapter, tick=2),
            cloud_request=cloud_request(
                cloud_payload,
                tick=2,
                num_steps=config.num_steps,
                seed=config.seed + 1,
            ),
        )

        link.set_connected(False)
        disconnected = await coordinator.infer_async(
            edge_request(resources.edge_adapter, tick=3),
            cloud_request=cloud_request(
                cloud_payload,
                tick=3,
                num_steps=config.num_steps,
                seed=config.seed + 2,
            ),
        )

        summary: dict[str, Any] = {
            "mode": config.failover.mode.value,
            "cloud_model_id": resources.cloud_engine.package.spec.model_id,
            "edge_model_id": resources.edge_engine.package.spec.model_id,
            "cloud_device": resources.cloud_session.device.device_id,
            "edge_device": resources.edge_session.device.device_id,
            "load_time_s": resources.load_time_s,
            "first_tick_source": first.source.value,
            "first_tick_time_s": first_tick_time_s,
            "cloud_in_flight_after_first": cloud_in_flight_after_first,
            "second_tick_source": second.source.value,
            "second_tick_cloud_sequence": second.source_sequence_id,
            "disconnected_tick_source": disconnected.source.value,
            "disconnected_reason": (
                disconnected.fallback_reason.value
                if disconnected.fallback_reason is not None
                else None
            ),
            "action_shape": tuple(actions(second.result).shape),
            "final_action_device": str(actions(second.result).device),
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(
                torch.device(config.cloud_device)
            )
            / (1024**2),
            "cloud_weight": config.cloud_weight,
        }
        if second.source is ResultSource.BLENDED:
            edge_actions = fusion_evidence["edge"]
            cloud_actions = fusion_evidence["cloud"]
            fused_actions = fusion_evidence["fused"]
            expected = edge_actions + (cloud_actions - edge_actions) * config.cloud_weight
            summary.update(
                {
                    "fusion_max_abs_error": float((fused_actions - expected).abs().max().item()),
                    "fused_vs_edge_max_abs": float(
                        (fused_actions - edge_actions).abs().max().item()
                    ),
                    "fused_vs_cloud_max_abs": float(
                        (fused_actions - cloud_actions).abs().max().item()
                    ),
                }
            )
        return summary
    finally:
        await coordinator.aclose()
        await asyncio.gather(
            resources.edge_engine.aclose(),
            resources.cloud_engine.aclose(),
        )
        resources._closed = True


def edge_request(adapter: Any, *, tick: int) -> InferenceRequest:
    action_dim = adapter.action_dim
    target = [tick * (index + 1) / action_dim for index in range(action_dim)]
    payload = adapter.preprocess_one(RawRequest(observation={"target": target}))
    return InferenceRequest(payload=payload, request_id=f"cpu-edge-{tick}")


def cloud_request(
    payload: Mapping[str, Any],
    *,
    tick: int,
    num_steps: int,
    seed: int,
) -> InferenceRequest:
    return InferenceRequest(
        payload=payload,
        request_id=f"gpu-cloud-{tick}",
        num_steps=num_steps,
        seed=seed,
    )


def actions(result: InferenceResult) -> Any:
    output = result.output
    return output["actions"] if isinstance(output, Mapping) else output


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - model extras require torch
        raise RuntimeError("the CPU/GPU collaboration demo requires PyTorch") from error
    return torch
