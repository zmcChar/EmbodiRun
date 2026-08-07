"""One get-coffee edge-only or oracle-plan trial."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embodied_runtime.distributed import PlanManager
from embodied_runtime.evaluation import ExperimentCondition, TrialRecord
from embodied_runtime.robots.action import RobotAction
from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.tasks.planning import PlanStepStatus

from .planning import activate_oracle_plan
from .scene import coffee_mug_is_placed
from .settings import GET_COFFEE_COMPOSITE_PROMPT, VLABenchPilotConfig


@dataclass(frozen=True, slots=True)
class EpisodeRun:
    record: TrialRecord
    events: tuple[dict[str, Any], ...]


def run_episode(
    *,
    config: VLABenchPilotConfig,
    endpoint: Any,
    runner: Any,
    condition: ExperimentCondition,
    seed: int,
    policy_seeder: Callable[[int], None],
    clock: Callable[[], float],
) -> EpisodeRun:
    observation: RobotObservation = endpoint.reset(seed=seed)
    fingerprint = endpoint.initial_fingerprint
    policy_seeder(seed)
    runner.reset_action_queue()

    plan_manager: PlanManager | None = None
    if condition is ExperimentCondition.ORACLE_PLAN:
        plan_manager = activate_oracle_plan(
            observation=observation,
            fingerprint=fingerprint,
            seed=seed,
        )
        plan_manager.record_feedback(PlanStepStatus.ACTIVE)

    pair_id = f"{config.task}-seed-{seed:08d}"
    edge_latencies_ms: list[float] = []
    chunk_generation_latencies_ms: list[float] = []
    queue_hit_latencies_ms: list[float] = []
    deadline_misses = 0
    events: list[dict[str, Any]] = []
    subgoals_completed = 0
    contained_streak = 0
    success = False
    steps = 0

    for step_index in range(config.max_episode_steps):
        if condition is ExperimentCondition.EDGE_ONLY:
            prompt = GET_COFFEE_COMPOSITE_PROMPT
            plan_step_id = None
        else:
            assert plan_manager is not None
            current_step = plan_manager.current_step
            if current_step is None:
                break
            prompt = current_step.instruction
            plan_step_id = current_step.step_id

        start_s = clock()
        selected = runner.select_action(observation.values, prompt)
        full_latency_s = clock() - start_s
        if full_latency_s < 0:
            raise RuntimeError("clock moved backwards while measuring edge inference")
        latency_ms = full_latency_s * 1000.0
        edge_latencies_ms.append(latency_ms)
        model_latency_ms = selected.inference_latency_s * 1000.0
        if bool(getattr(selected, "generated_chunk", False)):
            chunk_generation_latencies_ms.append(model_latency_ms)
        else:
            queue_hit_latencies_ms.append(model_latency_ms)
        if full_latency_s > config.control_period_s:
            deadline_misses += 1

        outcome = endpoint.step(
            RobotAction(
                timestamp_s=time.time(),
                values={"action": selected.action},
                metadata={
                    "condition": condition.value,
                    "prompt": prompt,
                    "plan_step_id": plan_step_id,
                },
            )
        )
        steps = step_index + 1
        observation = outcome.observation
        event = {
            "step": steps,
            "prompt": prompt,
            "plan_step_id": plan_step_id,
            "edge_latency_ms": latency_ms,
            "model_latency_ms": model_latency_ms,
            "generated_chunk": bool(getattr(selected, "generated_chunk", False)),
            "queue_remaining": getattr(selected, "queue_remaining", None),
            "success": bool(outcome.success),
            "physics_error": bool(outcome.info.get("physics_error", False)),
        }

        if outcome.success:
            success = True
            subgoals_completed = 2
            if plan_manager is not None and plan_manager.current_step is not None:
                plan_manager.advance(PlanStepStatus.SUCCEEDED)
            event["subgoal_completed"] = "press_start"
            events.append(event)
            break
        if outcome.done:
            events.append(event)
            break

        if subgoals_completed == 0:
            contained_streak = contained_streak + 1 if coffee_mug_is_placed(endpoint) else 0
        queue_remaining = getattr(selected, "queue_remaining", None)
        at_action_chunk_boundary = queue_remaining in (None, 0)
        if (
            subgoals_completed == 0
            and contained_streak >= config.subgoal_stability_steps
            and at_action_chunk_boundary
        ):
            subgoals_completed = 1
            event["subgoal_completed"] = "place_mug"
            event["contained_streak"] = contained_streak
            if plan_manager is not None:
                plan_manager.advance(PlanStepStatus.SUCCEEDED)
                if plan_manager.current_step is not None:
                    plan_manager.record_feedback(PlanStepStatus.ACTIVE)
                runner.reset_action_queue()

        events.append(event)
        if (
            condition is ExperimentCondition.ORACLE_PLAN
            and subgoals_completed == 0
            and steps == config.first_subgoal_budget
        ):
            events.append(
                {
                    "step": steps,
                    "event": "first_subgoal_budget_reached",
                    "plan_step_id": "place_mug",
                }
            )

    record = TrialRecord(
        pair_id=pair_id,
        task_id=config.task,
        seed=seed,
        condition=condition,
        success=success,
        steps=steps,
        edge_latencies_ms=tuple(edge_latencies_ms),
        planner_latencies_ms=(),
        chunk_generation_latencies_ms=tuple(chunk_generation_latencies_ms),
        queue_hit_latencies_ms=tuple(queue_hit_latencies_ms),
        deadline_misses=deadline_misses,
        subgoals_completed=subgoals_completed,
        subgoals_total=2,
        initial_fingerprint=fingerprint,
    )
    return EpisodeRun(record=record, events=tuple(events))
