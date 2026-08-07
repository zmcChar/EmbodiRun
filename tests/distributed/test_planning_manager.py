from __future__ import annotations

from dataclasses import replace

import pytest

from embodied_runtime.distributed.planning import AsyncPlannerEndpoint, PlanManager
from embodied_runtime.tasks.planning import PlanStep, PlanStepStatus

from ._planning_helpers import GatePlanner, plan, request


def test_planner_endpoint_protocol_is_runtime_checkable() -> None:
    assert isinstance(GatePlanner(), AsyncPlannerEndpoint)


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
    plan_request = request()
    manager = PlanManager("session-1", clock=lambda: now[0])

    with pytest.raises(ValueError, match=match):
        manager.offer_plan(replace(plan(plan_request), **change), request=plan_request)

    assert manager.pending_plan is None


def test_plan_manager_rejects_expired_stale_and_disallowed_plans() -> None:
    now = [201.0]
    plan_request = request()
    manager = PlanManager("session-1", clock=lambda: now[0])

    with pytest.raises(ValueError, match="already expired"):
        manager.offer_plan(plan(plan_request, expires_at_s=200.0), request=plan_request)

    now[0] = 110.0
    assert manager.offer_plan(plan(plan_request), request=plan_request)
    newer_request = request(request_id="request-2")
    with pytest.raises(ValueError, match="not newer"):
        manager.offer_plan(plan(newer_request), request=newer_request)

    disallowed = replace(
        plan(newer_request, revision=2),
        steps=(PlanStep(step_id="step-1", instruction="wave", skill="wave"),),
    )
    with pytest.raises(ValueError, match="disallowed skill"):
        manager.offer_plan(disallowed, request=newer_request)


def test_pending_plan_activates_atomically_only_at_safe_boundary() -> None:
    now = [110.0]
    plan_request = request()
    offered_plan = plan(plan_request)
    manager = PlanManager("session-1", clock=lambda: now[0])
    manager.offer_plan(offered_plan, request=plan_request)

    assert manager.activate_pending(safe_boundary=False) is None
    assert manager.active_plan is None
    assert manager.pending_plan is offered_plan

    assert manager.activate_pending(safe_boundary=True) is offered_plan
    assert manager.pending_plan is None
    assert manager.current_step == offered_plan.steps[0]

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
    plan_request = request()
    manager = PlanManager("session-1", clock=lambda: now[0])
    manager.offer_plan(plan(plan_request, expires_at_s=120.0), request=plan_request)

    now[0] = 121.0

    assert manager.activate_pending(safe_boundary=True) is None
    assert manager.pending_plan is None
    assert manager.active_plan is None


def test_pending_plan_optionally_rejects_stale_observation_at_activation() -> None:
    now = [110.0]
    plan_request = request()
    offered_plan = plan(plan_request)
    manager = PlanManager("session-1", clock=lambda: now[0])
    manager.offer_plan(offered_plan, request=plan_request)

    assert (
        manager.activate_pending(
            safe_boundary=False,
            current_observation_timestamp_s=101.0,
            max_observation_staleness_s=5.0,
        )
        is None
    )
    assert manager.pending_plan is offered_plan

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
    plan_request = request()
    offered_plan = plan(plan_request)
    manager = PlanManager("session-1", clock=lambda: 110.0)
    manager.offer_plan(offered_plan, request=plan_request)

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
        is offered_plan
    )
