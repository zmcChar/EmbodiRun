from __future__ import annotations

import pytest

from embodied_runtime.contracts.task import (
    PlanEnvelope,
    PlanFeedback,
    PlanRequest,
    PlanStep,
    PlanStepStatus,
    TaskGoal,
)


def test_task_contracts_normalize_immutable_sequences_and_status() -> None:
    goal = TaskGoal(
        task_id=" task-1 ",
        session_id=" session-1 ",
        instruction="  put the block in the bowl  ",
        allowed_skills=["pick", "place"],
        created_at_s=10.0,
    )
    feedback = PlanFeedback(
        task_id=goal.task_id,
        session_id=goal.session_id,
        plan_id="plan-1",
        revision=1,
        step_id="step-1",
        status="succeeded",
        timestamp_s=11.0,
    )
    request = PlanRequest(
        goal=goal,
        observation={"camera": "frame"},
        observation_id="frame-1",
        observation_timestamp_s=12.0,
        feedback=[feedback],
        requested_at_s=12.0,
        request_id="request-1",
    )
    plan = PlanEnvelope(
        request_id=request.request_id,
        task_id=goal.task_id,
        session_id=goal.session_id,
        revision=1,
        steps=[
            PlanStep(step_id="step-1", instruction="pick block", skill="pick"),
            PlanStep(step_id="step-2", instruction="place block", skill="place"),
        ],
        created_at_s=13.0,
        expires_at_s=30.0,
        based_on_observation_id=request.observation_id,
        plan_id="plan-1",
    )

    assert goal.task_id == "task-1"
    assert goal.instruction == "put the block in the bowl"
    assert goal.allowed_skills == ("pick", "place")
    assert request.feedback == (feedback,)
    assert feedback.status is PlanStepStatus.SUCCEEDED
    assert isinstance(plan.steps, tuple)
    assert plan.based_on_observation_id == "frame-1"


def test_plan_request_requires_traceable_observation_lineage() -> None:
    goal = TaskGoal(
        task_id="task-1",
        session_id="session-1",
        instruction="pick",
        created_at_s=1.0,
    )

    with pytest.raises(ValueError, match="requires observation_id"):
        PlanRequest(
            goal=goal,
            observation={"camera": "frame"},
            requested_at_s=2.0,
        )
    with pytest.raises(ValueError, match="provided together"):
        PlanRequest(
            goal=goal,
            observation_id="frame-1",
            requested_at_s=2.0,
        )


def test_plan_request_snapshots_observation_for_async_consumers() -> None:
    goal = TaskGoal(
        task_id="task-1",
        session_id="session-1",
        instruction="pick",
        created_at_s=1.0,
    )
    observation = {
        "camera": bytearray(b"frame"),
        "state": {"joints": [0.0, 1.0]},
    }

    request = PlanRequest(
        goal=goal,
        observation=observation,
        observation_id="frame-1",
        observation_timestamp_s=2.0,
        requested_at_s=2.0,
    )
    observation["camera"][0] = ord("X")
    observation["state"]["joints"][0] = 99.0

    assert request.observation == {
        "camera": bytearray(b"frame"),
        "state": {"joints": [0.0, 1.0]},
    }


def test_plan_request_rejects_cross_task_feedback() -> None:
    goal = TaskGoal(
        task_id="task-1",
        session_id="session-1",
        instruction="pick",
        created_at_s=1.0,
    )
    feedback = PlanFeedback(
        task_id="task-2",
        session_id="session-1",
        plan_id="plan-1",
        revision=1,
        step_id="step-1",
        status=PlanStepStatus.FAILED,
        timestamp_s=2.0,
    )

    with pytest.raises(ValueError, match="must match the request goal"):
        PlanRequest(
            goal=goal,
            feedback=(feedback,),
            requested_at_s=3.0,
        )


def test_plan_envelope_requires_unique_steps_and_valid_expiry() -> None:
    step = PlanStep(step_id="step-1", instruction="pick")
    common = {
        "request_id": "request-1",
        "task_id": "task-1",
        "session_id": "session-1",
        "revision": 1,
        "created_at_s": 10.0,
        "plan_id": "plan-1",
    }

    with pytest.raises(ValueError, match="step_id values must be unique"):
        PlanEnvelope(
            **common,
            steps=(step, step),
            expires_at_s=20.0,
        )
    with pytest.raises(ValueError, match="greater than created_at_s"):
        PlanEnvelope(
            **common,
            steps=(step,),
            expires_at_s=10.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"task_id": "", "session_id": "session-1", "instruction": "pick"},
        {"task_id": "task-1", "session_id": "session-1", "instruction": "  "},
        {
            "task_id": "task-1",
            "session_id": "session-1",
            "instruction": "pick",
            "allowed_skills": ("pick", "pick"),
        },
    ),
)
def test_task_goal_rejects_invalid_identity_text_and_skill_set(kwargs) -> None:
    with pytest.raises(ValueError):
        TaskGoal(**kwargs, created_at_s=1.0)
