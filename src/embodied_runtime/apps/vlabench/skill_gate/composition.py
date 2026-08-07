"""Paired privileged-executor planner skill-gate composition."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from embodied_runtime.simulators import (
    VLABenchSimulatorEndpoint,
    run_texas_holdem_privileged_skill_executor,
)

from ..texas_holdem_task import TEXAS_HOLDEM_TASK
from .planning import endpoint_fingerprint, inspect_planner_poker_deal
from .reporting import write_gate_report
from .settings import (
    PlannerPokerDeal,
    PlannerSkillCondition,
    PlannerSkillGateConfig,
    PlannerSkillTrial,
)
from .trial import run_condition


def run_planner_skill_gate(
    config: PlannerSkillGateConfig,
    *,
    cloud_planner: Any | None = None,
    endpoint_factory: Callable[..., Any] = VLABenchSimulatorEndpoint,
    deal_inspector: Callable[[Any], PlannerPokerDeal] | None = None,
    skill_executor: Callable[..., Any] = run_texas_holdem_privileged_skill_executor,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[PlannerSkillTrial, ...]:
    """Run paired plan conditions and persist after every completed seed."""

    if not isinstance(config, PlannerSkillGateConfig):
        raise TypeError("config must be a PlannerSkillGateConfig")
    deal_inspector = deal_inspector or inspect_planner_poker_deal
    conditions = [PlannerSkillCondition.WRONG_PLAN, PlannerSkillCondition.ORACLE_PLAN]
    if cloud_planner is not None:
        conditions.append(PlannerSkillCondition.CLOUD_PLAN)

    trials: list[PlannerSkillTrial] = []
    for pair_index, seed in enumerate(config.seeds):
        endpoint = endpoint_factory(
            TEXAS_HOLDEM_TASK,
            max_episode_steps=config.max_episode_steps,
            render_resolution=config.render_resolution,
        )
        expected_fingerprint: str | None = None
        expected_deal: PlannerPokerDeal | None = None
        try:
            ordered = tuple(
                conditions[(index + pair_index) % len(conditions)]
                for index in range(len(conditions))
            )
            for condition in ordered:
                observation = endpoint.reset(seed=seed)
                fingerprint = endpoint_fingerprint(endpoint)
                deal = deal_inspector(endpoint)
                if expected_fingerprint is None:
                    expected_fingerprint = fingerprint
                    expected_deal = deal
                elif fingerprint != expected_fingerprint or deal != expected_deal:
                    raise RuntimeError(
                        f"seed {seed} did not reproduce the paired initial state and deal"
                    )
                trials.append(
                    run_condition(
                        config=config,
                        endpoint=endpoint,
                        observation=observation,
                        fingerprint=fingerprint,
                        deal=deal,
                        seed=seed,
                        condition=condition,
                        cloud_planner=cloud_planner,
                        endpoint_factory=endpoint_factory,
                        skill_executor=skill_executor,
                        clock=clock,
                    )
                )
        finally:
            endpoint.close()
        write_gate_report(config, trials)
    return tuple(trials)


__all__ = ["run_planner_skill_gate"]
