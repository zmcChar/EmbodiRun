from __future__ import annotations

import asyncio

import pytest

from embodied_runtime.distributed.planning import AsyncPlanCoordinator, PlanManager
from embodied_runtime.tasks.planning import PlanRequest, PlanStepStatus

from ._planning_helpers import GatePlanner, NeverPlanner, async_test, plan, request


@async_test
async def test_coordinator_is_nonblocking_allows_one_request_and_delivers_result() -> None:
    plan_request = request()
    planner = GatePlanner()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    coordinator = AsyncPlanCoordinator(
        planner=planner,
        manager=manager,
        request_timeout_s=1.0,
    )

    assert coordinator.submit(plan_request)
    assert not coordinator.submit(request(request_id="request-2"))
    await asyncio.wait_for(planner.started.wait(), timeout=1.0)
    assert coordinator.request_in_flight
    assert manager.pending_plan is None

    planner.release.set()
    result = await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1.0)

    assert result == plan(plan_request)
    assert manager.pending_plan == result
    assert coordinator.last_error is None
    assert planner.calls == 1
    await coordinator.close()


@async_test
async def test_coordinator_enforces_real_timeout_and_records_failure() -> None:
    planner = NeverPlanner()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    coordinator = AsyncPlanCoordinator(
        planner=planner,
        manager=manager,
        request_timeout_s=0.01,
    )

    assert coordinator.submit(request())
    await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1.0)

    assert isinstance(coordinator.last_error, asyncio.TimeoutError)
    assert planner.cancelled.is_set()
    assert not coordinator.request_in_flight
    assert manager.pending_plan is None
    await coordinator.aclose()


@async_test
async def test_planner_timeout_does_not_evict_the_active_edge_plan() -> None:
    initial_request = request()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    initial_plan = plan(initial_request)
    manager.offer_plan(initial_plan, request=initial_request)
    manager.activate_pending(safe_boundary=True)
    coordinator = AsyncPlanCoordinator(
        planner=NeverPlanner(),
        manager=manager,
        request_timeout_s=0.01,
    )

    assert coordinator.submit(request(request_id="replan-timeout"))
    await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1.0)

    assert isinstance(coordinator.last_error, asyncio.TimeoutError)
    assert manager.active_plan is initial_plan
    assert manager.current_step is initial_plan.steps[0]
    assert manager.pending_plan is None
    await coordinator.close()


@async_test
async def test_new_revision_stays_pending_until_a_safe_control_boundary() -> None:
    initial_request = request()
    manager = PlanManager("session-1", clock=lambda: 110.0)
    initial_plan = plan(initial_request)
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
    planner = GatePlanner(revision=2)
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
    planner = NeverPlanner()
    coordinator = AsyncPlanCoordinator(
        planner=planner,
        manager=PlanManager("session-1", clock=lambda: 110.0),
        request_timeout_s=10.0,
    )
    coordinator.submit(request())
    await asyncio.wait_for(planner.started.wait(), timeout=1.0)

    await coordinator.close()

    assert planner.cancelled.is_set()
    assert not coordinator.request_in_flight
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.submit(request(request_id="request-2"))
