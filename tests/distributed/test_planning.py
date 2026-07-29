from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import wraps

import pytest

from embodied_runtime.contracts.task import (
    PlanEnvelope,
    PlanRequest,
    PlanStep,
    PlanStepStatus,
    TaskGoal,
)
from embodied_runtime.distributed.planning import (
    AsyncPlanCoordinator,
    AsyncPlannerEndpoint,
    PlanManager,
)


def async_test(function):
    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


def _request(
    *,
    request_id: str = "request-1",
    task_id: str = "task-1",
    session_id: str = "session-1",
) -> PlanRequest:
    return PlanRequest(
        goal=TaskGoal(
            task_id=task_id,
            session_id=session_id,
            instruction="put the block in the bowl",
            allowed_skills=("pick", "place"),
            created_at_s=90.0,
        ),
        observation={"camera": "frame"},
        observation_id="frame-1",
        observation_timestamp_s=95.0,
        requested_at_s=96.0,
        request_id=request_id,
    )


def _plan(
    request: PlanRequest,
    *,
    revision: int = 1,
    expires_at_s: float = 200.0,
) -> PlanEnvelope:
    return PlanEnvelope(
        request_id=request.request_id,
        task_id=request.goal.task_id,
        session_id=request.goal.session_id,
        revision=revision,
        steps=(
            PlanStep(step_id="step-1", instruction="pick block", skill="pick"),
            PlanStep(step_id="step-2", instruction="place block", skill="place"),
        ),
        created_at_s=100.0,
        expires_at_s=expires_at_s,
        based_on_observation_id=request.observation_id,
        plan_id=f"plan-{revision}",
    )


class _GatePlanner:
    def __init__(self, *, revision: int = 1) -> None:
        self.calls = 0
        self.revision = revision
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def plan_async(self, request: PlanRequest) -> PlanEnvelope:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return _plan(request, revision=self.revision)


class _NeverPlanner:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def plan_async(self, request: PlanRequest) -> PlanEnvelope:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


def test_planner_endpoint_protocol_is_runtime_checkable() -> None:
    assert isinstance(_GatePlanner(), AsyncPlannerEndpoint)


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"request_id": "other-request"}, "request_id"),
        ({"task_id": "other-task"}, "task_id"),
        ({"session_id": "other-session"}, "session"),
        ({"based_on_observation_id": "other-frame"}, "observation lineage"),
    ),
)
def test_plan_manager_rejects_unbound_planner_results(change, match: str) -> None:
    now = [110.0]
    request = _request()
    manager = PlanManager("session-1", clock=lambda: now[0])

    with pytest.raises(ValueError, match=match):
        manager.offer_plan(replace(_plan(request), **change), request=request)

    assert manager.pending_plan is None


def test_plan_manager_rejects_expired_stale_and_disallowed_plans() -> None:
    now = [201.0]
    request = _request()
    manager = PlanManager("session-1", clock=lambda: now[0])

    with pytest.raises(ValueError, match="already expired"):
        manager.offer_plan(_plan(request, expires_at_s=200.0), request=request)

    now[0] = 110.0
    assert manager.offer_plan(_plan(request), request=request)
    newer_request = _request(request_id="request-2")
    with pytest.raises(ValueError, match="not newer"):
        manager.offer_plan(_plan(newer_request), request=newer_request)

    disallowed = replace(
        _plan(newer_request, revision=2),
        steps=(PlanStep(step_id="step-1", instruction="wave", skill="wave"),),
    )
    with pytest.raises(ValueError, match="disallowed skill"):
        manager.offer_plan(disallowed, request=newer_request)


def test_pending_plan_activates_atomically_only_at_safe_boundary() -> None:
    now = [110.0]
    request = _request()
    plan = _plan(request)
    manager = PlanManager("session-1", clock=lambda: now[0])
    manager.offer_plan(plan, request=request)

    assert manager.activate_pending(safe_boundary=False) is None
    assert manager.active_plan is None
    assert manager.pending_plan is plan

    assert manager.activate_pending(safe_boundary=True) is plan
    assert manager.pending_plan is None
    assert manager.current_step == plan.steps[0]

    active = manager.record_feedback(PlanStepStatus.ACTIVE, timestamp_s=111.0)
    first = manager.advance(timestamp_s=112.0)
    second = manager.advance(PlanStepStatus.FAILED, timestamp_s=113.0)

    assert active.step_id == "step-1"
    assert first.status is PlanStepStatus.SUCCEEDED
    assert manager.current_step is None
    assert second.step_id == "step-2"
    assert manager.plan_complete
    assert manager.feedback_history == (active, first, second)


def test_pending_plan_is_discarded_if_it_expires_before_safe_boundary() -> None:
    now = [110.0]
    request = _request()
    manager = PlanManager("session-1", clock=lambda: now[0])
    manager.offer_plan(_plan(request, expires_at_s=120.0), request=request)

    now[0] = 121.0

    assert manager.activate_pending(safe_boundary=True) is None
    assert manager.pending_plan is None
    assert manager.active_plan is None


def test_pending_plan_optionally_rejects_stale_observation_at_activation() -> None:
    now = [110.0]
    request = _request()
    plan = _plan(request)
    manager = PlanManager("session-1", clock=lambda: now[0])
    manager.offer_plan(plan, request=request)

    assert (
        manager.activate_pending(
            safe_boundary=False,
            current_observation_timestamp_s=101.0,
            max_observation_staleness_s=5.0,
        )
        is None
    )
    assert manager.pending_plan is plan

    assert (
        manager.activate_pending(
            safe_boundary=True,
            current_observation_timestamp_s=101.0,
            max_observation_staleness_s=5.0,
        )
        is None
    )
    assert manager.pending_plan is None
    assert manager.active_plan is None


def test_pending_plan_accepts_fresh_observation_and_validates_staleness_arguments() -> None:
    request = _request()
    plan = _plan(request)
    manager = PlanManager("session-1", clock=lambda: 110.0)
    manager.offer_plan(plan, request=request)

    with pytest.raises(ValueError, match="provided together"):
        manager.activate_pending(
            safe_boundary=True,
            current_observation_timestamp_s=99.0,
        )
    with pytest.raises(ValueError, match="cannot precede"):
        manager.activate_pending(
            safe_boundary=True,
            current_observation_timestamp_s=94.0,
            max_observation_staleness_s=5.0,
        )

    assert (
        manager.activate_pending(
            safe_boundary=True,
            current_observation_timestamp_s=100.0,
            max_observation_staleness_s=5.0,
        )
        is plan
    )


@async_test
async def test_coordinator_is_nonblocking_allows_one_request_and_delivers_result() -> None:
    request = _request()
    planner = _GatePlanner()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    coordinator = AsyncPlanCoordinator(
        planner=planner,
        manager=manager,
        request_timeout_s=1.0,
    )

    assert coordinator.submit(request)
    assert not coordinator.submit(_request(request_id="request-2"))
    await asyncio.wait_for(planner.started.wait(), timeout=1.0)
    assert coordinator.request_in_flight
    assert manager.pending_plan is None

    planner.release.set()
    result = await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1.0)

    assert result == _plan(request)
    assert manager.pending_plan == result
    assert coordinator.last_error is None
    assert planner.calls == 1
    await coordinator.close()


@async_test
async def test_coordinator_enforces_real_timeout_and_records_failure() -> None:
    planner = _NeverPlanner()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    coordinator = AsyncPlanCoordinator(
        planner=planner,
        manager=manager,
        request_timeout_s=0.01,
    )

    assert coordinator.submit(_request())
    await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1.0)

    assert isinstance(coordinator.last_error, asyncio.TimeoutError)
    assert planner.cancelled.is_set()
    assert not coordinator.request_in_flight
    assert manager.pending_plan is None
    await coordinator.aclose()


@async_test
async def test_planner_timeout_does_not_evict_the_active_edge_plan() -> None:
    initial_request = _request()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    initial_plan = _plan(initial_request)
    manager.offer_plan(initial_plan, request=initial_request)
    manager.activate_pending(safe_boundary=True)
    coordinator = AsyncPlanCoordinator(
        planner=_NeverPlanner(),
        manager=manager,
        request_timeout_s=0.01,
    )

    assert coordinator.submit(_request(request_id="replan-timeout"))
    await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1.0)

    assert isinstance(coordinator.last_error, asyncio.TimeoutError)
    assert manager.active_plan is initial_plan
    assert manager.current_step is initial_plan.steps[0]
    assert manager.pending_plan is None
    await coordinator.close()


@async_test
async def test_new_revision_stays_pending_until_a_safe_control_boundary() -> None:
    initial_request = _request()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    initial_plan = _plan(initial_request)
    manager.offer_plan(initial_plan, request=initial_request)
    manager.activate_pending(safe_boundary=True)
    manager.record_feedback(PlanStepStatus.ACTIVE, timestamp_s=111.0)
    replan_request = PlanRequest(
        goal=initial_request.goal,
        observation={"camera": "frame-2"},
        observation_id="frame-2",
        observation_timestamp_s=112.0,
        active_plan_id=initial_plan.plan_id,
        active_revision=initial_plan.revision,
        feedback=manager.feedback_history,
        requested_at_s=113.0,
        request_id="request-revision-2",
    )
    planner = _GatePlanner(revision=2)
    coordinator = AsyncPlanCoordinator(
        planner=planner,
        manager=manager,
        request_timeout_s=1.0,
    )

    assert coordinator.submit(replan_request)
    await asyncio.wait_for(planner.started.wait(), timeout=1.0)
    assert manager.current_step is initial_plan.steps[0]
    planner.release.set()
    revised = await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1.0)

    assert revised is not None
    assert revised.revision == 2
    assert manager.pending_plan is revised
    assert manager.active_plan is initial_plan
    assert manager.activate_pending(safe_boundary=False) is None
    assert manager.active_plan is initial_plan
    assert manager.activate_pending(safe_boundary=True) is revised
    assert manager.current_step is revised.steps[0]
    await coordinator.close()


@async_test
async def test_close_cancels_owned_request_and_rejects_new_work() -> None:
    planner = _NeverPlanner()
    coordinator = AsyncPlanCoordinator(
        planner=planner,
        manager=PlanManager("session-1", clock=lambda: 110.0),
        request_timeout_s=10.0,
    )
    coordinator.submit(_request())
    await asyncio.wait_for(planner.started.wait(), timeout=1.0)

    await coordinator.close()

    assert planner.cancelled.is_set()
    assert not coordinator.request_in_flight
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.submit(_request(request_id="request-2"))
