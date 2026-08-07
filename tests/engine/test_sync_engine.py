from __future__ import annotations

import time

import pytest

from embodied_runtime.backends import MemoryStats
from embodied_runtime.engine import (
    EngineConfig,
    EnginePayloadError,
    ExecutionEngine,
    InferenceRequest,
    MemoryBudgetExceededError,
    RequestDeadlineExceededError,
)
from embodied_runtime.models.request import RawRequest

from .fakes import (
    FAKE_FORWARD_PACKAGE_ID,
    FakeSession,
    make_forward_package,
    make_package,
)


def test_sync_flow_executes_plan_only_through_backend_session() -> None:
    session = FakeSession()
    engine = ExecutionEngine(make_package(num_steps=4), session)

    result = engine.infer(2.5, request_id="sync")

    assert result.request_id == "sync"
    assert result.output == {"actions": pytest.approx(2.5)}
    assert [name for name, _, _ in session.calls] == [
        "encode_prefix",
        "init_state",
        "denoise_step",
        "advance_state",
        "denoise_step",
        "advance_state",
        "denoise_step",
        "advance_state",
        "denoise_step",
        "advance_state",
        "finalize",
    ]
    snapshot = engine.metrics.snapshot()
    assert snapshot.submitted == 1
    assert snapshot.succeeded == 1
    assert snapshot.stage_calls == 11


def test_package_may_declare_step_returns_next_state() -> None:
    session = FakeSession(step_returns_state=True)
    engine = ExecutionEngine(
        make_package(num_steps=3, step_returns_state=True),
        session,
    )

    result = engine.infer(99.0)

    assert result.output == {"actions": pytest.approx(3.0)}


def test_sync_deadline_is_relative_to_request_creation() -> None:
    session = FakeSession()
    engine = ExecutionEngine(make_package(), session)
    request = InferenceRequest(
        payload=1.0,
        deadline_s=0.01,
        created_at_s=time.monotonic() - 1.0,
    )

    with pytest.raises(RequestDeadlineExceededError):
        engine.infer(request)

    assert session.calls == []
    assert engine.metrics.snapshot().deadline_exceeded == 1


def test_raw_request_must_be_preprocessed_before_engine() -> None:
    engine = ExecutionEngine(make_package(), FakeSession())

    with pytest.raises(EnginePayloadError, match="model adapter"):
        engine.infer(RawRequest(observation={"image": object()}))


def test_memory_budget_controls_admission_without_allocating_memory() -> None:
    session = FakeSession(
        memory_stats=MemoryStats(
            allocated_bytes=60,
            reserved_bytes=80,
            total_bytes=100,
            free_bytes=20,
        )
    )
    engine = ExecutionEngine(
        make_package(),
        session,
        EngineConfig(memory_budget_bytes=64),
    )

    with pytest.raises(MemoryBudgetExceededError, match="80 > 64"):
        engine.infer(1.0)

    assert session.calls == []


def test_close_is_idempotent_and_closes_session() -> None:
    session = FakeSession()
    engine = ExecutionEngine(make_package(), session)

    engine.close()
    engine.close()

    assert session.closed


def test_engine_rejects_session_from_another_package() -> None:
    with pytest.raises(ValueError, match="different ModelPackage"):
        ExecutionEngine(
            make_package(),
            FakeSession(package_id="another-package"),
        )


def test_single_forward_plan_submits_once_without_state_update() -> None:
    session = FakeSession(package_id=FAKE_FORWARD_PACKAGE_ID)
    engine = ExecutionEngine(make_forward_package(), session)

    result = engine.infer({"features": 3.0}, request_id="forward")

    assert result.output == {"features": 3.0}
    assert [name for name, _, _ in session.calls] == ["forward"]
    assert engine.metrics.snapshot().stage_calls == 1


def test_single_forward_plan_rejects_num_steps() -> None:
    session = FakeSession(package_id=FAKE_FORWARD_PACKAGE_ID)
    engine = ExecutionEngine(make_forward_package(), session)

    with pytest.raises(EnginePayloadError, match="num_steps"):
        engine.infer(1.0, num_steps=2)

    assert session.calls == []
