from __future__ import annotations

import asyncio

from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FallbackReason,
    ResultSource,
)
from embodied_runtime.distributed.communication import DummyLink, DummyRemoteInferenceEndpoint

from ._failover_helpers import GateEndpoint, ImmediateEndpoint, async_test


@async_test
async def test_disconnect_immediately_invalidates_cloud_authority() -> None:
    edge = ImmediateEndpoint("edge")
    cloud = ImmediateEndpoint("cloud")
    link = DummyLink()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud, link),
    )

    await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    await asyncio.wait_for(coordinator.wait_for_cloud_idle(), timeout=1.0)
    connected = await coordinator.infer_async("edge-2", cloud_request="cloud-2")
    assert connected.source is ResultSource.CLOUD

    link.set_connected(False)
    disconnected = await coordinator.infer_async("edge-3", cloud_request="cloud-3")

    assert disconnected.result == "edge:edge-3"
    assert disconnected.source is ResultSource.EDGE
    assert disconnected.fallback_reason is FallbackReason.CLOUD_DISCONNECTED
    await coordinator.aclose()


@async_test
async def test_late_cloud_result_cannot_preempt_newer_sequence() -> None:
    edge = ImmediateEndpoint("edge")
    cloud = GateEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(
            max_cloud_sequence_lag=0,
            cloud_submit_interval_s=1000.0,
        ),
    )

    await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    await coordinator.infer_async("edge-2", cloud_request="cloud-2")
    cloud.release.set()
    await asyncio.wait_for(coordinator.wait_for_cloud_idle(), timeout=1.0)
    decision = await coordinator.infer_async("edge-3", cloud_request="cloud-3")

    assert decision.result == "edge:edge-3"
    assert decision.source is ResultSource.EDGE
    assert decision.fallback_reason is FallbackReason.STALE_CLOUD_RESULT
    await coordinator.aclose()


@async_test
async def test_cloud_ttl_includes_time_spent_in_inference() -> None:
    edge = ImmediateEndpoint("edge")
    cloud = GateEndpoint()
    now = [0.0]
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(
            cloud_result_ttl_s=1.0,
            max_cloud_sequence_lag=10,
            cloud_submit_interval_s=1000.0,
        ),
        clock=lambda: now[0],
    )

    await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    now[0] = 2.0
    cloud.release.set()
    await asyncio.wait_for(coordinator.wait_for_cloud_idle(), timeout=1.0)
    decision = await coordinator.infer_async("edge-2", cloud_request="cloud-2")

    assert decision.source is ResultSource.EDGE
    assert decision.fallback_reason is FallbackReason.STALE_CLOUD_RESULT
    await coordinator.aclose()


@async_test
async def test_hung_cloud_request_times_out_without_blocking_edge() -> None:
    edge = ImmediateEndpoint("edge")
    cloud = GateEndpoint()
    now = [0.0]
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(cloud_request_timeout_s=0.5),
        clock=lambda: now[0],
    )

    first = await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    assert first.fallback_reason is FallbackReason.CLOUD_PENDING
    now[0] = 1.0
    second = await coordinator.infer_async("edge-2", cloud_request="cloud-2")

    assert second.result == "edge:edge-2"
    assert second.source is ResultSource.EDGE
    assert second.fallback_reason is FallbackReason.CLOUD_TIMEOUT
    await coordinator.aclose()


@async_test
async def test_wait_for_cloud_idle_observes_real_request_timeout() -> None:
    edge = ImmediateEndpoint("edge")
    cloud = GateEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(cloud_request_timeout_s=0.01),
    )

    await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    await asyncio.wait_for(coordinator.wait_for_cloud_idle(), timeout=0.5)
    decision = await coordinator.infer_async("edge-2", cloud_request="cloud-2")

    assert decision.source is ResultSource.EDGE
    assert decision.fallback_reason is FallbackReason.CLOUD_TIMEOUT
    await coordinator.aclose()


@async_test
async def test_reconnect_requires_a_fresh_epoch_result_before_cloud_recovers() -> None:
    edge = ImmediateEndpoint("edge")
    cloud = ImmediateEndpoint("cloud")
    link = DummyLink()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud, link),
    )

    await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    await coordinator.wait_for_cloud_idle()
    assert (
        await coordinator.infer_async("edge-2", cloud_request="cloud-2")
    ).source is ResultSource.CLOUD

    link.set_connected(False)
    assert (
        await coordinator.infer_async("edge-3", cloud_request="cloud-3")
    ).source is ResultSource.EDGE
    link.set_connected(True)
    recovering = await coordinator.infer_async("edge-4", cloud_request="cloud-4")
    assert recovering.source is ResultSource.EDGE

    await coordinator.wait_for_cloud_idle()
    recovered = await coordinator.infer_async("edge-5", cloud_request="cloud-5")
    assert recovered.result == "cloud:cloud-4"
    assert recovered.source is ResultSource.CLOUD
    assert recovered.connection_epoch == 2
    await coordinator.aclose()
