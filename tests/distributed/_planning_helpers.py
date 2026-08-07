from __future__ import annotations

import asyncio
from functools import wraps

from embodied_runtime.tasks.planning import PlanEnvelope, PlanRequest, PlanStep, TaskGoal


def async_test(function):
    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


def request(
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


def plan(
    plan_request: PlanRequest,
    *,
    revision: int = 1,
    expires_at_s: float = 200.0,
) -> PlanEnvelope:
    return PlanEnvelope(
        request_id=plan_request.request_id,
        task_id=plan_request.goal.task_id,
        session_id=plan_request.goal.session_id,
        revision=revision,
        steps=(
            PlanStep(step_id="step-1", instruction="pick block", skill="pick"),
            PlanStep(step_id="step-2", instruction="place block", skill="place"),
        ),
        created_at_s=100.0,
        expires_at_s=expires_at_s,
        based_on_observation_id=plan_request.observation_id,
        plan_id=f"plan-{revision}",
    )


class GatePlanner:
    def __init__(self, *, revision: int = 1) -> None:
        self.calls = 0
        self.revision = revision
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def plan_async(self, plan_request: PlanRequest) -> PlanEnvelope:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return plan(plan_request, revision=self.revision)


class NeverPlanner:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def plan_async(self, plan_request: PlanRequest) -> PlanEnvelope:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")
