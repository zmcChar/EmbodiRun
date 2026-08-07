from __future__ import annotations

import asyncio
from functools import wraps

import pytest

from embodied_runtime.engine import EngineState, ExecutionEngine, RequestCancelledError
from embodied_runtime.engine.execution_engine import (
    EngineState as DirectEngineState,
)
from embodied_runtime.engine.execution_engine import (
    ExecutionEngine as DirectExecutionEngine,
)

from .fakes import FakeSession, make_package


def async_test(function):
    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


def test_public_execution_engine_import_path_is_stable() -> None:
    assert DirectExecutionEngine is ExecutionEngine
    assert DirectEngineState is EngineState


@async_test
async def test_stopped_engine_can_restart_with_a_fresh_worker() -> None:
    session = FakeSession()
    engine = ExecutionEngine(make_package(num_steps=1), session)

    await engine.start()
    first = await engine.infer_async(1.0, request_id="first-run")
    await engine.stop()

    assert first.output == {"actions": pytest.approx(1.0)}
    assert engine.state is EngineState.STOPPED
    assert not session.closed

    await engine.start()
    second = await engine.infer_async(2.0, request_id="second-run")
    await engine.aclose()

    assert second.output == {"actions": pytest.approx(2.0)}
    assert engine.state is EngineState.CLOSED
    assert session.closed


@async_test
async def test_active_request_id_is_unique_but_reusable_after_delivery() -> None:
    session = FakeSession(block_first_encode=True)
    engine = ExecutionEngine(make_package(num_steps=1), session)

    first = engine.submit(1.0, request_id="reusable")
    assert await asyncio.to_thread(session.first_encode_started.wait, 1.0)
    with pytest.raises(ValueError, match="already active"):
        engine.submit(2.0, request_id="reusable")

    session.release_first_encode.set()
    await first
    second = engine.submit(3.0, request_id="reusable")
    assert (await second).output == {"actions": pytest.approx(3.0)}
    await engine.aclose()


@async_test
async def test_stop_without_drain_cancels_running_and_queued_work() -> None:
    session = FakeSession(
        step_delay_s=0.03,
        honor_context_cancellation=True,
    )
    engine = ExecutionEngine(make_package(num_steps=8), session)
    running = engine.submit(1.0, request_id="running")
    assert await asyncio.to_thread(session.step_started.wait, 1.0)
    queued = engine.submit(2.0, request_id="queued")

    await engine.stop(drain=False)
    with pytest.raises(RequestCancelledError):
        await running
    with pytest.raises(RequestCancelledError):
        await queued

    assert engine.state is EngineState.STOPPED
    assert engine.metrics.snapshot().cancelled == 2
    await engine.aclose()
