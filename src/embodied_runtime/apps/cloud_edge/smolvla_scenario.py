"""Asynchronous failover scenario and result reporting."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverDecision,
    FailoverMode,
)
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult

from .pi05_codec import PI05_ACTION_DIM, PI05_ACTION_HORIZON, action_shape
from .smolvla_runtime import (
    EdgeRuntime,
    Pi05TcpEndpoint,
    build_cloud_endpoint,
    build_edge_runtime,
)
from .smolvla_settings import SmolVLAPi05AsyncConfig


def run_smolvla_pi05_async(
    config: SmolVLAPi05AsyncConfig,
    *,
    edge_runtime_factory: Callable[[SmolVLAPi05AsyncConfig], EdgeRuntime] | None = None,
    cloud_endpoint_factory: Callable[[SmolVLAPi05AsyncConfig], Pi05TcpEndpoint] | None = None,
) -> dict[str, Any]:
    """Build the edge model once and execute the three-tick async scenario."""

    build_edge = edge_runtime_factory or build_edge_runtime
    build_cloud = cloud_endpoint_factory or build_cloud_endpoint
    resources = build_edge(config)
    try:
        cloud = build_cloud(config)
        return asyncio.run(run_async_scenario(resources, cloud, config))
    except BaseException:
        resources.close()
        raise


async def run_async_scenario(
    resources: EdgeRuntime,
    cloud: Pi05TcpEndpoint,
    config: SmolVLAPi05AsyncConfig,
) -> dict[str, Any]:
    coordinator = AsyncFailoverCoordinator(
        edge=resources.engine,
        cloud=cloud,
        config=config.failover,
    )
    try:
        await resources.engine.start()

        first, first_time_s = await timed_tick(coordinator, resources, config, tick=1)
        cloud_in_flight_after_first = coordinator.cloud_request_in_flight

        cloud_wait_started = time.perf_counter()
        if config.failover.mode is not FailoverMode.EDGE_ONLY:
            # This wait is deliberately outside the control-tick critical path.
            await asyncio.wait_for(
                coordinator.wait_for_cloud_idle(),
                timeout=config.failover.cloud_request_timeout_s + 5.0,
            )
        outside_control_cloud_wait_time_s = time.perf_counter() - cloud_wait_started

        second, second_time_s = await timed_tick(coordinator, resources, config, tick=2)

        # Changing the endpoint epoch immediately invalidates cached cloud authority.
        cloud.set_connected(False)
        disconnected, disconnected_time_s = await timed_tick(coordinator, resources, config, tick=3)

        edge_spec = resources.engine.package.spec
        return {
            "status": "ok",
            "mode": config.failover.mode.value,
            "edge_model_id": edge_spec.model_id,
            "edge_device": resources.session.device.device_id,
            "edge_dtype_policy": "preserve",
            "edge_load_time_s": resources.load_time_s,
            "cloud_transport": "length_prefixed_json_v1",
            "cloud_host": config.cloud_host,
            "cloud_port": config.cloud_port,
            "cloud_in_flight_after_first": cloud_in_flight_after_first,
            "outside_control_cloud_wait_time_s": outside_control_cloud_wait_time_s,
            "first_tick": decision_record(first, first_time_s),
            "second_tick": decision_record(second, second_time_s),
            "disconnected_tick": decision_record(disconnected, disconnected_time_s),
            "action_contracts": {
                "edge_smolvla": {
                    "shape": [edge_spec.action_horizon, edge_spec.action_dim],
                    "semantics": "smolvla_robot_specific_action_space",
                },
                "cloud_pi05": {
                    "shape": [PI05_ACTION_HORIZON, PI05_ACTION_DIM],
                    "semantics": "pi05_policy_action_space",
                },
                "semantically_compatible": False,
            },
            "numeric_blend_performed": False,
            "input_mode": "independent_synthetic_payloads",
            "control_path_contract": (
                "each tick awaits edge inference only; cloud completion is consumed "
                "opportunistically or awaited outside the control path"
            ),
        }
    finally:
        await coordinator.aclose()
        await resources.engine.aclose()
        resources._closed = True


async def timed_tick(
    coordinator: AsyncFailoverCoordinator,
    resources: EdgeRuntime,
    config: SmolVLAPi05AsyncConfig,
    *,
    tick: int,
) -> tuple[FailoverDecision[InferenceResult], float]:
    started = time.perf_counter()
    decision = await coordinator.infer_async(
        InferenceRequest(
            payload=resources.payload,
            request_id=f"smolvla-edge-{tick}",
            num_steps=config.num_steps,
            seed=config.seed + tick - 1,
        ),
        cloud_request=InferenceRequest(
            payload={},
            request_id=f"pi05-cloud-{tick}",
            num_steps=config.num_steps,
            seed=config.seed + tick - 1,
        ),
    )
    return decision, time.perf_counter() - started


def decision_record(
    decision: FailoverDecision[InferenceResult],
    control_path_time_s: float,
) -> dict[str, Any]:
    return {
        "source": decision.source.value,
        "sequence_id": decision.sequence_id,
        "source_sequence_id": decision.source_sequence_id,
        "connection_epoch": decision.connection_epoch,
        "fallback_reason": (
            decision.fallback_reason.value if decision.fallback_reason is not None else None
        ),
        "cloud_result_age_s": decision.cloud_result_age_s,
        "selected_action_shape": list(action_shape(decision.result.output)),
        "selected_model_id": decision.result.metadata.get("model_id"),
        "selected_execution_time_s": decision.result.execution_time_s,
        "control_path_time_s": control_path_time_s,
    }
