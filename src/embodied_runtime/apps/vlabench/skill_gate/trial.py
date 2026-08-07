"""One privileged-executor planner skill-gate condition."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..texas_holdem_task import TEXAS_HOLDEM_TASK
from .planning import (
    activate_and_extract_cloud_selection,
    build_cloud_plan_request,
    call_cloud_planner,
    clock_value,
    deterministic_wrong_card_selection,
)
from .settings import (
    PlannerPokerDeal,
    PlannerSkillCondition,
    PlannerSkillGateConfig,
    PlannerSkillTrial,
)


def run_condition(
    *,
    config: PlannerSkillGateConfig,
    endpoint: Any,
    observation: Any,
    fingerprint: str,
    deal: PlannerPokerDeal,
    seed: int,
    condition: PlannerSkillCondition,
    cloud_planner: Any | None,
    endpoint_factory: Callable[..., Any],
    skill_executor: Callable[..., Any],
    clock: Callable[[], float],
) -> PlannerSkillTrial:
    pair_id = f"{TEXAS_HOLDEM_TASK}-seed-{seed:08d}"
    selected: tuple[str, ...] = ()
    planner_latency_ms = 0.0
    planner_accepted = True
    error: Exception | None = None

    if condition is PlannerSkillCondition.ORACLE_PLAN:
        selected = deal.target_card_names
    elif condition is PlannerSkillCondition.WRONG_PLAN:
        selected = deterministic_wrong_card_selection(deal, seed=seed)
    else:
        if cloud_planner is None:
            raise RuntimeError("cloud_plan condition requires a cloud planner")
        request = build_cloud_plan_request(
            deal=deal,
            seed=seed,
            fingerprint=fingerprint,
            observation_timestamp_s=observation.timestamp_s,
        )
        started = clock_value(clock)
        try:
            envelope = call_cloud_planner(cloud_planner, request)
            planner_latency_ms = (clock_value(clock) - started) * 1000.0
            selected = activate_and_extract_cloud_selection(envelope, request)
        except Exception as caught:  # noqa: BLE001 - model/transport errors are result data
            planner_latency_ms = max(0.0, (clock_value(clock) - started) * 1000.0)
            planner_accepted = False
            error = caught

    if not planner_accepted:
        return PlannerSkillTrial(
            pair_id=pair_id,
            seed=seed,
            condition=condition,
            initial_fingerprint=fingerprint,
            hand_type=deal.hand_type,
            true_target_card_names=deal.target_card_names,
            selected_card_names=selected,
            planner_accepted=False,
            planner_latency_ms=planner_latency_ms,
            execution_attempted=False,
            success=False,
            waypoint_steps=0,
            settle_steps=0,
            execution_label=None,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

    def shadow_factory(task: str, *, max_episode_steps: int) -> Any:
        return endpoint_factory(
            task,
            max_episode_steps=max_episode_steps,
            render_resolution=config.render_resolution,
        )

    try:
        replay = skill_executor(
            endpoint,
            seed=seed,
            selected_card_names=selected,
            settle_repeats=config.settle_repeats,
            shadow_endpoint_factory=shadow_factory,
            shadow_max_episode_steps=config.shadow_max_episode_steps,
        )
    except Exception as caught:  # noqa: BLE001 - simulator failures are result data
        return PlannerSkillTrial(
            pair_id=pair_id,
            seed=seed,
            condition=condition,
            initial_fingerprint=fingerprint,
            hand_type=deal.hand_type,
            true_target_card_names=deal.target_card_names,
            selected_card_names=selected,
            planner_accepted=True,
            planner_latency_ms=planner_latency_ms,
            execution_attempted=True,
            success=False,
            waypoint_steps=0,
            settle_steps=0,
            execution_label=None,
            error_type=type(caught).__name__,
            error_message=str(caught),
        )
    return PlannerSkillTrial(
        pair_id=pair_id,
        seed=seed,
        condition=condition,
        initial_fingerprint=fingerprint,
        hand_type=deal.hand_type,
        true_target_card_names=deal.target_card_names,
        selected_card_names=selected,
        planner_accepted=True,
        planner_latency_ms=planner_latency_ms,
        execution_attempted=True,
        success=bool(replay.success),
        waypoint_steps=int(replay.waypoint_steps),
        settle_steps=int(replay.settle_steps),
        execution_label=str(replay.trajectory.evaluation_label),
    )
