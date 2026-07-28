from __future__ import annotations

import asyncio
from functools import wraps

from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FailoverMode,
    FallbackReason,
    ResultSource,
)
from embodied_runtime.distributed.communication import (
    DummyLink,
    DummyRemoteInferenceEndpoint,
)


def async_test(function):
    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


class _ImmediateEndpoint:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.calls = 0

    async def infer_async(self, request: str) -> str:
        self.calls += 1
        return f"{self.prefix}:{request}"


class _GateEndpoint:
    def __init__(self, prefix: str = "cloud") -> None:
        self.prefix = prefix
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer_async(self, request: str) -> str:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return f"{self.prefix}:{request}"


class _FailIfCalledEndpoint:
    def __init__(self) -> None:
        self.calls = 0

    async def infer_async(self, request: str) -> str:
        self.calls += 1
        raise AssertionError(f"cloud should not receive {request!r}")


@async_test
async def test_blocked_cloud_never_blocks_edge_tick() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
    )

    tick = asyncio.create_task(coordinator.infer_async("edge-1", cloud_request="cloud-1"))
    await asyncio.wait_for(cloud.started.wait(), timeout=1.0)
    decision = await asyncio.wait_for(tick, timeout=1.0)

    assert decision.result == "edge:edge-1"
    assert decision.source is ResultSource.EDGE
    assert decision.fallback_reason is FallbackReason.CLOUD_PENDING
    assert coordinator.cloud_request_in_flight
    await coordinator.aclose()


@async_test
async def test_fresh_cloud_result_preempts_output_but_edge_stays_hot() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(max_cloud_sequence_lag=1),
    )

    first = await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    assert first.source is ResultSource.EDGE
    await asyncio.wait_for(cloud.started.wait(), timeout=1.0)
    cloud.release.set()
    await asyncio.wait_for(coordinator.wait_for_cloud_idle(), timeout=1.0)

    second = await coordinator.infer_async("edge-2", cloud_request="cloud-2")

    assert second.result == "cloud:cloud-1"
    assert second.source is ResultSource.CLOUD
    assert second.source_sequence_id == 1
    assert edge.calls == 2
    await coordinator.aclose()


@async_test
async def test_disconnect_immediately_invalidates_cloud_authority() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _ImmediateEndpoint("cloud")
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
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
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
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
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
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
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
async def test_edge_only_mode_never_starts_cloud_work() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _FailIfCalledEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(mode=FailoverMode.EDGE_ONLY),
    )

    decision = await coordinator.infer_async("edge-1", cloud_request="cloud-1")

    assert decision.source is ResultSource.EDGE
    assert decision.fallback_reason is FallbackReason.CLOUD_DISABLED
    assert cloud.calls == 0
    assert not coordinator.cloud_request_in_flight
    await coordinator.aclose()


@async_test
async def test_reconnect_requires_a_fresh_epoch_result_before_cloud_recovers() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _ImmediateEndpoint("cloud")
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
