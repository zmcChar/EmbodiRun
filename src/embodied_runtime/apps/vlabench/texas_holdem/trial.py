"""One matched Texas Hold'em condition trial."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embodied_runtime.distributed import PlanManager
from embodied_runtime.evaluation import ExperimentCondition, TrialRecord
from embodied_runtime.robots.action import RobotAction
from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.tasks.planning import PlanStepStatus

from .place_controller import EndpointPlaceController
from .planning import (
    activate_card_plan,
    advance_completed_plan_steps,
    deal_document,
    wrong_plan_cards,
)
from .scene import (
    contained_card_names,
    contained_target_names,
    grasped_poker_name,
    inspect_poker_deal,
)
from .settings import (
    TEXAS_HOLDEM_COMPOSITE_PROMPT,
    PokerCard,
    PokerDeal,
    TexasHoldemExperimentConfig,
)


@dataclass(frozen=True, slots=True)
class EpisodeRun:
    record: TrialRecord
    deal: PokerDeal
    events: tuple[dict[str, Any], ...]


def run_episode(
    *,
    config: TexasHoldemExperimentConfig,
    endpoint: Any,
    runner: Any,
    place_controller: EndpointPlaceController,
    condition: ExperimentCondition,
    seed: int,
    policy_seeder: Callable[[int], None],
    clock: Callable[[], float],
) -> EpisodeRun:
    observation: RobotObservation = endpoint.reset(seed=seed)
    fingerprint = endpoint.initial_fingerprint
    deal = inspect_poker_deal(endpoint)
    policy_seeder(seed)
    runner.reset_action_queue()

    plan_manager: PlanManager | None = None
    planned_cards: tuple[PokerCard, ...] = ()
    planner_name: str | None = None
    planner_latencies_ms: list[float] = []
    if condition in {ExperimentCondition.ORACLE_PLAN, ExperimentCondition.WRONG_PLAN}:
        planner_start_s = clock()
        if condition is ExperimentCondition.ORACLE_PLAN:
            planned_cards = deal.targets
            planner_name = "simulator_oracle"
        else:
            planned_cards = wrong_plan_cards(deal, seed=seed)
            planner_name = "deterministic_wrong_control"
        plan_manager = activate_card_plan(
            observation=observation,
            fingerprint=fingerprint,
            seed=seed,
            deal=deal,
            cards=planned_cards,
            planner_name=planner_name,
        )
        planner_latency_s = clock() - planner_start_s
        if planner_latency_s < 0:
            raise RuntimeError("clock moved backwards while measuring oracle planning")
        planner_latencies_ms.append(planner_latency_s * 1000.0)
        plan_manager.record_feedback(PlanStepStatus.ACTIVE)

    pair_id = f"{config.task}-seed-{seed:08d}"
    edge_latencies_ms: list[float] = []
    chunk_generation_latencies_ms: list[float] = []
    queue_hit_latencies_ms: list[float] = []
    deadline_misses = 0
    events: list[dict[str, Any]] = [{"event": "deal", **deal_document(deal)}]
    if plan_manager is not None:
        events.append(
            {
                "event": "plan_activated",
                "planner": planner_name,
                "planned_card_names": [card.name for card in planned_cards],
                "planned_prompts": [card.primitive_prompt for card in planned_cards],
                "deliberately_incorrect": (condition is ExperimentCondition.WRONG_PLAN),
            }
        )
    completed_targets: frozenset[str] = frozenset()
    completed_plan_cards: frozenset[str] = frozenset()
    placement_index = 0
    success = False
    steps = 0

    while steps < config.max_episode_steps:
        if condition is ExperimentCondition.EDGE_ONLY:
            prompt = TEXAS_HOLDEM_COMPOSITE_PROMPT
            plan_step_id = None
        else:
            assert plan_manager is not None
            advance_completed_plan_steps(plan_manager, completed_plan_cards)
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
        model_latency_s = float(selected.inference_latency_s)
        if not math.isfinite(model_latency_s) or model_latency_s < 0:
            raise RuntimeError("runner returned an invalid inference latency")
        model_latency_ms = model_latency_s * 1000.0
        generated_chunk = bool(getattr(selected, "generated_chunk", False))
        if generated_chunk:
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
                    "controller": "edge_policy",
                },
            )
        )
        steps += 1
        observation = outcome.observation
        events.append(
            {
                "event": "edge_policy_step",
                "step": steps,
                "prompt": prompt,
                "plan_step_id": plan_step_id,
                "edge_latency_ms": latency_ms,
                "model_latency_ms": model_latency_ms,
                "generated_chunk": generated_chunk,
                "queue_remaining": getattr(selected, "queue_remaining", None),
                "success": bool(outcome.success),
                "physics_error": bool(outcome.info.get("physics_error", False)),
            }
        )
        if outcome.success:
            success = True
            completed_targets = frozenset(deal.target_names)
            break
        if outcome.done:
            break

        grasped_poker = grasped_poker_name(endpoint)
        if grasped_poker is not None:
            controller_result = place_controller.execute(
                endpoint=endpoint,
                observation=observation,
                available_steps=config.max_episode_steps - steps,
                condition=condition,
                prompt=prompt,
                grasped_poker=grasped_poker,
                placement_index=placement_index,
            )
            placement_index += 1
            steps += controller_result.steps
            observation = controller_result.observation
            first_controller_step = steps - controller_result.steps + 1
            for offset, event in enumerate(controller_result.events):
                events.append({"step": first_controller_step + offset, **event})
            runner.reset_action_queue()
            if controller_result.outcome is not None:
                if controller_result.outcome.success:
                    success = True
                    completed_targets = frozenset(deal.target_names)
                    break
                if controller_result.outcome.done:
                    break

        completed_targets = contained_target_names(endpoint, deal)
        if plan_manager is not None:
            planned_names = tuple(card.name for card in planned_cards)
            completed_plan_cards = contained_card_names(endpoint, planned_names)
            advance_completed_plan_steps(plan_manager, completed_plan_cards)

    record = TrialRecord(
        pair_id=pair_id,
        task_id=config.task,
        seed=seed,
        condition=condition,
        success=success,
        steps=steps,
        edge_latencies_ms=tuple(edge_latencies_ms),
        planner_latencies_ms=tuple(planner_latencies_ms),
        chunk_generation_latencies_ms=tuple(chunk_generation_latencies_ms),
        queue_hit_latencies_ms=tuple(queue_hit_latencies_ms),
        deadline_misses=deadline_misses,
        subgoals_completed=len(completed_targets),
        subgoals_total=len(deal.target_names),
        initial_fingerprint=fingerprint,
    )
    return EpisodeRun(record=record, deal=deal, events=tuple(events))
