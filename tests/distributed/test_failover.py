from __future__ import annotations

import asyncio
from functools import wraps

import pytest

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


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("cloud_request_timeout_s", float("nan")),
        ("cloud_request_timeout_s", float("inf")),
        ("cloud_result_ttl_s", float("nan")),
        ("cloud_result_ttl_s", float("inf")),
        ("cloud_submit_interval_s", float("nan")),
        ("cloud_submit_interval_s", float("inf")),
    ),
)
def test_failover_config_rejects_nonfinite_timing(name: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        FailoverConfig(**{name: value})


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
async def test_cloud_factory_runs_only_for_an_admitted_submission() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
    )
    snapshots: list[str] = []

    def snapshot_first() -> str:
        snapshots.append("cloud-1")
        return "cloud-1"

    def snapshot_second() -> str:
        snapshots.append("cloud-2")
        return "cloud-2"

    await coordinator.infer_async(
        "edge-1",
        cloud_request_factory=snapshot_first,
    )
    await coordinator.infer_async(
        "edge-2",
        cloud_request_factory=snapshot_second,
    )

    assert snapshots == ["cloud-1"]
    await coordinator.aclose()


@async_test
async def test_cloud_factory_failure_keeps_edge_authoritative() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _FailIfCalledEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
    )

    def fail_snapshot() -> str:
        raise RuntimeError("camera buffer cannot be cloned")

    decision = await coordinator.infer_async(
        "edge-1",
        cloud_request_factory=fail_snapshot,
    )

    assert decision.result == "edge:edge-1"
    assert decision.source is ResultSource.EDGE
    assert decision.fallback_reason is FallbackReason.CLOUD_ERROR
    assert edge.calls == 1
    assert cloud.calls == 0
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


def test_async_blend_requires_an_explicit_result_fuser() -> None:
    with pytest.raises(ValueError, match="requires a result fuser"):
        AsyncFailoverCoordinator(
            edge=_ImmediateEndpoint("edge"),
            cloud=DummyRemoteInferenceEndpoint(_ImmediateEndpoint("cloud")),
            config=FailoverConfig(mode=FailoverMode.ASYNC_BLEND),
        )


@async_test
async def test_async_blend_combines_current_edge_and_fresh_cloud_results() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(mode=FailoverMode.ASYNC_BLEND),
        fuser=lambda edge_result, cloud_result: f"{edge_result}+{cloud_result}",
    )

    first = await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    assert first.source is ResultSource.EDGE
    cloud.release.set()
    await asyncio.wait_for(coordinator.wait_for_cloud_idle(), timeout=1.0)

    second = await coordinator.infer_async("edge-2", cloud_request="cloud-2")

    assert second.result == "edge:edge-2+cloud:cloud-1"
    assert second.source is ResultSource.BLENDED
    assert second.sequence_id == 2
    assert second.source_sequence_id == 1
    assert edge.calls == 2
    await coordinator.aclose()


@async_test
async def test_blend_failure_keeps_edge_result_authoritative() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _ImmediateEndpoint("cloud")

    def fail_to_blend(edge_result: str, cloud_result: str) -> str:
        raise ValueError(f"cannot blend {edge_result!r} with {cloud_result!r}")

    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud),
        config=FailoverConfig(mode=FailoverMode.ASYNC_BLEND),
        fuser=fail_to_blend,
    )
    await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    await coordinator.wait_for_cloud_idle()

    decision = await coordinator.infer_async("edge-2", cloud_request="cloud-2")

    assert decision.result == "edge:edge-2"
    assert decision.source is ResultSource.EDGE
    assert decision.fallback_reason is FallbackReason.BLEND_ERROR
    await coordinator.aclose()


@async_test
async def test_blend_disconnect_and_reconnect_require_a_fresh_cloud_epoch() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _ImmediateEndpoint("cloud")
    link = DummyLink()
    fused_pairs: list[tuple[str, str]] = []

    def record_blend(edge_result: str, cloud_result: str) -> str:
        fused_pairs.append((edge_result, cloud_result))
        return f"{edge_result}+{cloud_result}"

    coordinator = AsyncFailoverCoordinator(
        edge=edge,
        cloud=DummyRemoteInferenceEndpoint(cloud, link),
        config=FailoverConfig(mode=FailoverMode.ASYNC_BLEND),
        fuser=record_blend,
    )
    await coordinator.infer_async("edge-1", cloud_request="cloud-1")
    await coordinator.wait_for_cloud_idle()
    assert (
        await coordinator.infer_async("edge-2", cloud_request="cloud-2")
    ).source is ResultSource.BLENDED

    link.set_connected(False)
    disconnected = await coordinator.infer_async("edge-3", cloud_request="cloud-3")
    assert disconnected.source is ResultSource.EDGE
    assert disconnected.fallback_reason is FallbackReason.CLOUD_DISCONNECTED
    assert len(fused_pairs) == 1

    link.set_connected(True)
    recovering = await coordinator.infer_async("edge-4", cloud_request="cloud-4")
    assert recovering.source is ResultSource.EDGE
    await coordinator.wait_for_cloud_idle()
    recovered = await coordinator.infer_async("edge-5", cloud_request="cloud-5")
    assert recovered.source is ResultSource.BLENDED
    assert recovered.connection_epoch == 2
    assert fused_pairs[-1] == ("edge:edge-5", "cloud:cloud-4")
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
async def test_wait_for_cloud_idle_observes_real_request_timeout() -> None:
    edge = _ImmediateEndpoint("edge")
    cloud = _GateEndpoint()
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
