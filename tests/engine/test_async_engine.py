from __future__ import annotations

import asyncio
from functools import wraps

import pytest

from embodied_runtime.engine import (
    EngineConfig,
    ExecutionEngine,
    InferenceRequest,
    QueueFullError,
    RequestCancelledError,
    RequestDeadlineExceededError,
)

from .fakes import (
    FAKE_FORWARD_PACKAGE_ID,
    FakeSession,
    make_forward_package,
    make_package,
)


def async_test(function):
    """Keep the contract suite runnable even without the optional pytest plugin."""

    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


@async_test
async def test_dynamic_batch_runs_one_plan_and_splits_results() -> None:
    session = FakeSession()
    engine = ExecutionEngine(
        make_package(),
        session,
        EngineConfig(max_batch_size=3, max_wait_ms=20),
        batcher=lambda payloads: list(payloads),
    )

    results = await asyncio.gather(
        engine.infer_async(1.0, request_id="one"),
        engine.infer_async(2.0, request_id="two"),
        engine.infer_async(3.0, request_id="three"),
    )
    await engine.aclose()

    assert [result.output for result in results] == [
        {"actions": pytest.approx(1.0)},
        {"actions": pytest.approx(2.0)},
        {"actions": pytest.approx(3.0)},
    ]
    assert [name for name, _, _ in session.calls].count("encode_prefix") == 1
    snapshot = engine.metrics.snapshot()
    assert snapshot.batches == 1
    assert snapshot.succeeded == 3


@async_test
async def test_explicit_seeds_are_not_dynamically_batched() -> None:
    session = FakeSession()
    engine = ExecutionEngine(
        make_package(),
        session,
        EngineConfig(max_batch_size=2, max_wait_ms=10),
        batcher=lambda payloads: list(payloads),
    )

    await asyncio.gather(
        engine.infer_async(1.0, request_id="seed-one", seed=7),
        engine.infer_async(2.0, request_id="seed-two", seed=7),
    )
    await engine.aclose()

    assert [name for name, _, _ in session.calls].count("encode_prefix") == 2
    assert engine.metrics.snapshot().batches == 2


@async_test
async def test_priority_orders_requests_that_are_still_queued() -> None:
    session = FakeSession(block_first_encode=True)
    engine = ExecutionEngine(
        make_package(num_steps=1),
        session,
        EngineConfig(max_queue_size=4),
    )

    first = engine.submit(0.0, request_id="first", priority=0)
    assert await asyncio.to_thread(session.first_encode_started.wait, 1.0)
    low = engine.submit(1.0, request_id="low", priority=1)
    high = engine.submit(2.0, request_id="high", priority=10)
    session.release_first_encode.set()

    await asyncio.gather(first, low, high)
    await engine.aclose()

    encode_order = [
        request_ids[0] for name, request_ids, _ in session.calls if name == "encode_prefix"
    ]
    assert encode_order == ["first", "high", "low"]


@async_test
async def test_bounded_queue_applies_backpressure() -> None:
    session = FakeSession(block_first_encode=True)
    engine = ExecutionEngine(
        make_package(num_steps=1),
        session,
        EngineConfig(max_queue_size=1),
    )

    running = engine.submit(0.0, request_id="running")
    assert await asyncio.to_thread(session.first_encode_started.wait, 1.0)
    queued = engine.submit(1.0, request_id="queued")
    with pytest.raises(QueueFullError):
        engine.submit(2.0, request_id="rejected")

    session.release_first_encode.set()
    await asyncio.gather(running, queued)
    await engine.aclose()

    assert engine.metrics.snapshot().rejected == 1


@async_test
async def test_queued_request_can_be_cancelled() -> None:
    session = FakeSession(block_first_encode=True)
    engine = ExecutionEngine(make_package(num_steps=1), session)

    running = engine.submit(0.0, request_id="running")
    assert await asyncio.to_thread(session.first_encode_started.wait, 1.0)
    cancelled = engine.submit(1.0, request_id="cancelled")
    assert cancelled.cancel()
    session.release_first_encode.set()

    await running
    with pytest.raises(RequestCancelledError):
        await cancelled
    assert cancelled.status.value == "cancelled"
    await engine.aclose()

    assert engine.metrics.snapshot().cancelled == 1


@async_test
async def test_running_request_cancels_at_denoise_safe_point() -> None:
    session = FakeSession(
        step_delay_s=0.03,
        honor_context_cancellation=True,
    )
    engine = ExecutionEngine(make_package(num_steps=8), session)
    handle = engine.submit(1.0, request_id="cancel-running")

    assert await asyncio.to_thread(session.step_started.wait, 1.0)
    assert handle.cancel()
    with pytest.raises(RequestCancelledError):
        await handle
    await engine.aclose()

    step_calls = sum(name == "denoise_step" for name, _, _ in session.calls)
    assert step_calls < 8


@async_test
async def test_deadline_is_observed_between_denoise_steps() -> None:
    session = FakeSession(step_delay_s=0.03)
    engine = ExecutionEngine(make_package(num_steps=8), session)
    request = InferenceRequest(
        payload=1.0,
        request_id="deadline",
        deadline_s=0.01,
    )

    with pytest.raises(RequestDeadlineExceededError):
        await engine.infer_async(request)
    await engine.aclose()

    step_calls = sum(name == "denoise_step" for name, _, _ in session.calls)
    assert step_calls < 8
    assert engine.metrics.snapshot().deadline_exceeded == 1


@async_test
async def test_async_context_closes_backend_session() -> None:
    session = FakeSession()

    async with ExecutionEngine(make_package(num_steps=1), session) as engine:
        result = await engine.infer_async(4.0)
        assert result.output == {"actions": pytest.approx(4.0)}

    assert session.closed


@async_test
async def test_single_forward_dynamic_batch_submits_one_stage() -> None:
    session = FakeSession(package_id=FAKE_FORWARD_PACKAGE_ID)
    engine = ExecutionEngine(
        make_forward_package(),
        session,
        EngineConfig(max_batch_size=3, max_wait_ms=20),
        batcher=lambda payloads: list(payloads),
    )

    results = await asyncio.gather(
        engine.infer_async(1.0, request_id="forward-one"),
        engine.infer_async(2.0, request_id="forward-two"),
        engine.infer_async(3.0, request_id="forward-three"),
    )
    await engine.aclose()

    assert [result.output for result in results] == [1.0, 2.0, 3.0]
    assert [name for name, _, _ in session.calls] == ["forward"]
