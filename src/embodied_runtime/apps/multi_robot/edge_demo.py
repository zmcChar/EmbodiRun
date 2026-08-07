"""Composition and observable records for the multi-robot edge demo."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from embodied_runtime.distributed.communication import (
    MultiTenantTcpEndpoint,
    TcpJsonRequestClient,
    json_compatible,
)
from embodied_runtime.distributed.session import RobotSessionIdentity

from .._local_provider import build_local_provider
from .edge_runtime import RobotEdgeRuntime
from .edge_settings import MultiRobotEdgeConfig


def run_multi_robot_edge(config: MultiRobotEdgeConfig) -> list[dict[str, Any]]:
    if config.identity.session_id == "auto":
        config = replace(
            config,
            identity=RobotSessionIdentity(
                robot_id=config.identity.robot_id,
                edge_node_id=config.identity.edge_node_id,
                session_id=uuid.uuid4().hex,
                embodiment=config.identity.embodiment,
                action_space_id=config.identity.action_space_id,
            ),
        )
    edge = build_local_provider(config.provider)
    capabilities = edge.capabilities
    device = capabilities.device
    cloud = MultiTenantTcpEndpoint(
        TcpJsonRequestClient(
            config.cloud_host,
            config.cloud_port,
            timeout_s=config.cloud_timeout_s,
        ),
        config.identity,
        registration_metadata={
            "physical_host_id": config.physical_host_id,
            "physical_resource_id": config.physical_resource_id,
            "edge_provider": capabilities.name,
            "edge_provider_runtime": capabilities.runtime,
            "edge_model_id": capabilities.model.model_id,
            "edge_model_family": capabilities.model.family,
            "edge_action_dim": capabilities.model.action_dim,
            "edge_action_horizon": capabilities.model.action_horizon,
            "edge_backend": device.backend if device is not None else None,
            "edge_device": device.device_id if device is not None else None,
        },
    )
    runtime = RobotEdgeRuntime(
        identity=config.identity,
        edge=edge,
        cloud=cloud,
        failover=config.failover,
        registration_timeout_s=config.registration_timeout_s,
        reconnect_interval_s=config.reconnect_interval_s,
    )
    return asyncio.run(run_edge_demo(runtime, config))


async def run_edge_demo(
    runtime: RobotEdgeRuntime,
    config: MultiRobotEdgeConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        await runtime.start()
        for tick in range(1, config.ticks + 1):
            if tick == config.disconnect_tick:
                runtime.set_cloud_enabled(False)
            if tick == config.reconnect_tick:
                runtime.set_cloud_enabled(True)
            decision = await runtime.infer_observation(
                build_demo_observation(runtime, config, tick=tick),
                sequence_id=tick,
            )
            records.append(
                {
                    "robot_id": config.identity.robot_id,
                    "edge_node_id": config.identity.edge_node_id,
                    "session_id": config.identity.session_id,
                    "tick": tick,
                    "cloud_connected": runtime.cloud.connected,
                    "connection_epoch": decision.connection_epoch,
                    "source": decision.source.value,
                    "source_sequence_id": decision.source_sequence_id,
                    "cloud_result_age_s": decision.cloud_result_age_s,
                    "fallback_reason": (
                        decision.fallback_reason.value
                        if decision.fallback_reason is not None
                        else None
                    ),
                    "selected_observation_id": decision.result.metadata.get("observation_id"),
                    "selected_queue_time_s": decision.result.queue_time_s,
                    "selected_execution_time_s": decision.result.execution_time_s,
                    "output": json_compatible(decision.result.output, path="decision.output"),
                }
            )
            await asyncio.sleep(config.control_period_s if config.control_period_s else 0)
        if runtime.failover.cloud_request_in_flight:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    runtime.wait_for_cloud_idle(),
                    timeout=config.failover.cloud_request_timeout_s,
                )
        return records
    finally:
        await runtime.aclose()


def build_demo_observation(
    runtime: RobotEdgeRuntime,
    config: MultiRobotEdgeConfig,
    *,
    tick: int,
) -> Mapping[str, Any]:
    if config.observation_mode == "target_vector":
        action_dim = runtime.edge.capabilities.model.action_dim
        if action_dim is None or action_dim <= 0:
            raise ValueError("target-vector demo requires an edge model action_dim")
        return {
            "target": [
                config.observation_offset + float(tick) + index / action_dim
                for index in range(action_dim)
            ]
        }

    adapter = getattr(runtime.edge, "adapter", None)
    synthetic_batch = getattr(adapter, "synthetic_batch", None)
    if not callable(synthetic_batch):
        raise TypeError(
            "adapter_synthetic observations require runtime.edge.adapter.synthetic_batch"
        )
    observation = synthetic_batch(
        batch_size=1,
        language_length=config.observation_language_length,
        seed=config.observation_seed + tick - 1,
    )
    if not isinstance(observation, Mapping):
        raise TypeError("adapter synthetic_batch must return an observation mapping")
    return observation


__all__ = ["build_demo_observation", "run_edge_demo", "run_multi_robot_edge"]
