"""Privileged VLABench orchestration for simulator-only controlled trials.

The executor is privileged regardless of who selected the cards. A trajectory
is an oracle upper bound only when its selected cards exactly equal the
simulator's ground-truth best hand. Every main-environment transition is routed
through the simulator endpoint by the replay module.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from .vlabench_privileged_generation import (
    convert_vlabench_expert_waypoint,
    generate_texas_holdem_privileged_skill_trajectory,
)
from .vlabench_privileged_inspection import inspect_texas_holdem_deal
from .vlabench_privileged_replay import replay_privileged_skill_trajectory
from .vlabench_privileged_values import (
    DEFAULT_FINGER_TOLERANCE,
    DEFAULT_GRIPPER_OPEN_THRESHOLD,
    SIMULATOR_ORACLE_UPPER_BOUND,
    SIMULATOR_PRIVILEGED_PLAN_CONTROL,
    SIMULATOR_PRIVILEGED_SKILL_EXECUTOR,
    TEXAS_HOLDEM_TASK,
    PrivilegedOracleReplay,
    PrivilegedOracleTrajectory,
    PrivilegedOracleWaypoint,
    PrivilegedSkillReplay,
    PrivilegedSkillTrajectory,
    PrivilegedSkillWaypoint,
    TexasHoldemDealIdentity,
)

_DEFAULT_GRIPPER_OPEN_THRESHOLD = DEFAULT_GRIPPER_OPEN_THRESHOLD
_DEFAULT_FINGER_TOLERANCE = DEFAULT_FINGER_TOLERANCE


def run_texas_holdem_privileged_skill_executor(
    main_endpoint: Any,
    *,
    seed: int,
    selected_card_names: Sequence[str],
    settle_repeats: int = 3,
    shadow_endpoint_factory: Callable[..., Any] | None = None,
    shadow_max_episode_steps: int = 2000,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Generate selected-card skills in shadow and replay them on main."""

    trajectory = generate_texas_holdem_privileged_skill_trajectory(
        main_endpoint,
        seed=seed,
        selected_card_names=selected_card_names,
        shadow_endpoint_factory=shadow_endpoint_factory,
        shadow_max_episode_steps=shadow_max_episode_steps,
    )
    return replay_privileged_skill_trajectory(
        main_endpoint,
        trajectory,
        settle_repeats=settle_repeats,
        action_clock=action_clock,
    )


def generate_texas_holdem_privileged_oracle_trajectory(
    main_endpoint: Any,
    *,
    seed: int,
    target_card_names: Sequence[str],
    shadow_endpoint_factory: Callable[..., Any] | None = None,
    shadow_max_episode_steps: int = 2000,
) -> PrivilegedSkillTrajectory:
    """Generate a strict oracle trajectory and reject any non-oracle plan."""

    trajectory = generate_texas_holdem_privileged_skill_trajectory(
        main_endpoint,
        seed=seed,
        selected_card_names=target_card_names,
        shadow_endpoint_factory=shadow_endpoint_factory,
        shadow_max_episode_steps=shadow_max_episode_steps,
    )
    if not trajectory.is_oracle_upper_bound:
        raise RuntimeError(
            "privileged oracle entry point requires all and only the true target cards"
        )
    return trajectory


def replay_privileged_oracle_trajectory(
    main_endpoint: Any,
    trajectory: PrivilegedSkillTrajectory,
    *,
    settle_repeats: int = 3,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Replay a strict oracle trajectory and reject non-oracle labeling."""

    if not isinstance(trajectory, PrivilegedSkillTrajectory):
        raise TypeError("trajectory must be a PrivilegedSkillTrajectory")
    if not trajectory.is_oracle_upper_bound:
        raise RuntimeError("oracle replay cannot label a non-oracle selected-card plan")
    return replay_privileged_skill_trajectory(
        main_endpoint,
        trajectory,
        settle_repeats=settle_repeats,
        action_clock=action_clock,
    )


def run_texas_holdem_privileged_oracle_upper_bound(
    main_endpoint: Any,
    *,
    seed: int,
    target_card_names: Sequence[str],
    settle_repeats: int = 3,
    shadow_endpoint_factory: Callable[..., Any] | None = None,
    shadow_max_episode_steps: int = 2000,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Generate and replay the strict privileged oracle upper bound."""

    trajectory = generate_texas_holdem_privileged_oracle_trajectory(
        main_endpoint,
        seed=seed,
        target_card_names=target_card_names,
        shadow_endpoint_factory=shadow_endpoint_factory,
        shadow_max_episode_steps=shadow_max_episode_steps,
    )
    return replay_privileged_oracle_trajectory(
        main_endpoint,
        trajectory,
        settle_repeats=settle_repeats,
        action_clock=action_clock,
    )


__all__ = [
    "SIMULATOR_ORACLE_UPPER_BOUND",
    "SIMULATOR_PRIVILEGED_PLAN_CONTROL",
    "SIMULATOR_PRIVILEGED_SKILL_EXECUTOR",
    "TEXAS_HOLDEM_TASK",
    "PrivilegedOracleReplay",
    "PrivilegedOracleTrajectory",
    "PrivilegedOracleWaypoint",
    "PrivilegedSkillReplay",
    "PrivilegedSkillTrajectory",
    "PrivilegedSkillWaypoint",
    "TexasHoldemDealIdentity",
    "convert_vlabench_expert_waypoint",
    "generate_texas_holdem_privileged_oracle_trajectory",
    "generate_texas_holdem_privileged_skill_trajectory",
    "inspect_texas_holdem_deal",
    "replay_privileged_oracle_trajectory",
    "replay_privileged_skill_trajectory",
    "run_texas_holdem_privileged_oracle_upper_bound",
    "run_texas_holdem_privileged_skill_executor",
]
