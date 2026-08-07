"""Oracle get-coffee plan activation."""

from __future__ import annotations

import time

from embodied_runtime.distributed import PlanManager
from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.tasks.planning import PlanEnvelope, PlanRequest, TaskGoal

from .settings import (
    GET_COFFEE_COMPOSITE_PROMPT,
    GET_COFFEE_ORACLE_STEPS,
    GET_COFFEE_TASK,
)


def activate_oracle_plan(
    *,
    observation: RobotObservation,
    fingerprint: str,
    seed: int,
) -> PlanManager:
    session_id = f"get-coffee-{seed}"
    goal = TaskGoal(
        task_id=GET_COFFEE_TASK,
        session_id=session_id,
        instruction=GET_COFFEE_COMPOSITE_PROMPT,
        allowed_skills=("place_mug", "press_start"),
        metadata={"planner": "oracle", "seed": seed},
    )
    request = PlanRequest(
        goal=goal,
        observation=observation.values,
        observation_id=fingerprint,
        observation_timestamp_s=observation.timestamp_s,
    )
    created_at_s = time.time()
    envelope = PlanEnvelope(
        request_id=request.request_id,
        task_id=goal.task_id,
        session_id=goal.session_id,
        revision=1,
        steps=GET_COFFEE_ORACLE_STEPS,
        created_at_s=created_at_s,
        expires_at_s=created_at_s + 3600.0,
        based_on_observation_id=fingerprint,
        metadata={"planner": "oracle"},
    )
    manager = PlanManager(session_id)
    manager.offer_plan(envelope, request=request)
    if manager.activate_pending(safe_boundary=True) is None:
        raise RuntimeError("oracle plan could not be activated at episode reset")
    return manager
