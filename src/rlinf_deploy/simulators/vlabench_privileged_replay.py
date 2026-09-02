"""Replay privileged VLABench trajectories through the main endpoint only."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from rlinf_deploy.robots.adapter import RobotAction

from .base import EpisodeStep
from .vlabench_privileged_inspection import (
    endpoint_fingerprint,
    inspect_texas_holdem_deal,
    validate_endpoint_seed_if_available,
)
from .vlabench_privileged_values import (
    SIMULATOR_PRIVILEGED_SKILL_EXECUTOR,
    PrivilegedSkillReplay,
    PrivilegedSkillTrajectory,
    PrivilegedSkillWaypoint,
)


def replay_privileged_skill_trajectory(
    main_endpoint: Any,
    trajectory: PrivilegedSkillTrajectory,
    *,
    settle_repeats: int = 3,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Replay privileged waypoints through ``main_endpoint.step`` only."""

    if not isinstance(trajectory, PrivilegedSkillTrajectory):
        raise TypeError("trajectory must be a PrivilegedSkillTrajectory")
    if isinstance(settle_repeats, bool) or not isinstance(settle_repeats, int):
        raise TypeError("settle_repeats must be an integer")
    if settle_repeats < 0:
        raise ValueError("settle_repeats must be non-negative")
    if not callable(action_clock):
        raise TypeError("action_clock must be callable")

    current_fingerprint = endpoint_fingerprint(main_endpoint)
    if current_fingerprint != trajectory.initial_fingerprint:
        raise RuntimeError("main endpoint fingerprint changed before privileged oracle replay")
    if inspect_texas_holdem_deal(main_endpoint) != trajectory.deal:
        raise RuntimeError("main endpoint poker deal changed before privileged oracle replay")
    validate_endpoint_seed_if_available(main_endpoint, trajectory.seed)

    final_outcome: EpisodeStep | None = None
    waypoint_steps = 0
    settle_steps = 0
    for waypoint in trajectory.waypoints:
        final_outcome = _step_main_endpoint(
            main_endpoint,
            waypoint,
            phase="expert_waypoint",
            repeat_index=None,
            evaluation_label=trajectory.evaluation_label,
            is_oracle_upper_bound=trajectory.is_oracle_upper_bound,
            selected_card_names=trajectory.selected_card_names,
            action_clock=action_clock,
        )
        waypoint_steps += 1
        if final_outcome.done:
            break

    if final_outcome is None or not final_outcome.done:
        final_waypoint = trajectory.waypoints[-1]
        for repeat_index in range(settle_repeats):
            final_outcome = _step_main_endpoint(
                main_endpoint,
                final_waypoint,
                phase="final_settle",
                repeat_index=repeat_index,
                evaluation_label=trajectory.evaluation_label,
                is_oracle_upper_bound=trajectory.is_oracle_upper_bound,
                selected_card_names=trajectory.selected_card_names,
                action_clock=action_clock,
            )
            settle_steps += 1
            if final_outcome.done:
                break

    return PrivilegedSkillReplay(
        trajectory=trajectory,
        waypoint_steps=waypoint_steps,
        settle_steps=settle_steps,
        success=bool(final_outcome.success) if final_outcome is not None else False,
        terminated=bool(final_outcome.terminated) if final_outcome is not None else False,
        truncated=bool(final_outcome.truncated) if final_outcome is not None else False,
        final_outcome=final_outcome,
    )


def _step_main_endpoint(
    endpoint: Any,
    waypoint: PrivilegedSkillWaypoint,
    *,
    phase: str,
    repeat_index: int | None,
    evaluation_label: str,
    is_oracle_upper_bound: bool,
    selected_card_names: tuple[str, ...],
    action_clock: Callable[[], float],
) -> EpisodeStep:
    action = RobotAction(
        timestamp_s=float(action_clock()),
        values={"action": waypoint.action},
        metadata={
            "controller": SIMULATOR_PRIVILEGED_SKILL_EXECUTOR,
            "evaluation_label": evaluation_label,
            "oracle_upper_bound": is_oracle_upper_bound,
            "privileged_simulator_state": True,
            "deployable": False,
            "selected_card_names": selected_card_names,
            "phase": phase,
            "skill_index": waypoint.skill_index,
            "waypoint_index": waypoint.waypoint_index,
            "repeat_index": repeat_index,
        },
    )
    outcome = endpoint.step(action)
    if not isinstance(outcome, EpisodeStep):
        raise TypeError("main simulator endpoint step() must return EpisodeStep")
    return outcome


__all__ = ["replay_privileged_skill_trajectory"]
