"""Immutable values and validation for privileged VLABench experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .base import EpisodeStep

TEXAS_HOLDEM_TASK = "texas_holdem"
SIMULATOR_PRIVILEGED_SKILL_EXECUTOR = "simulator_privileged_skill_executor"
SIMULATOR_ORACLE_UPPER_BOUND = "simulator_oracle_privileged_upper_bound"
SIMULATOR_PRIVILEGED_PLAN_CONTROL = "simulator_privileged_plan_control"
DEFAULT_GRIPPER_OPEN_THRESHOLD = 0.03
DEFAULT_FINGER_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class TexasHoldemDealIdentity:
    """Stable simulator-visible identity used to validate a paired deal."""

    card_names: tuple[str, ...]
    target_card_names: tuple[str, ...]
    hand_type: str

    def __post_init__(self) -> None:
        card_names = non_empty_unique_names(self.card_names, field_name="card_names")
        target_card_names = non_empty_unique_names(
            self.target_card_names,
            field_name="target_card_names",
        )
        missing = set(target_card_names).difference(card_names)
        if missing:
            raise ValueError(f"target cards are missing from the deal: {sorted(missing)}")
        if not isinstance(self.hand_type, str) or not self.hand_type.strip():
            raise ValueError("hand_type must be a non-empty string")
        object.__setattr__(self, "card_names", card_names)
        object.__setattr__(self, "target_card_names", target_card_names)
        object.__setattr__(self, "hand_type", self.hand_type.strip())


@dataclass(frozen=True, slots=True)
class PrivilegedSkillWaypoint:
    """One deployable-shaped action generated from a privileged 8D waypoint."""

    action: tuple[float, float, float, float, float, float, float]
    skill_index: int
    waypoint_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, tuple) or len(self.action) != 7:
            raise ValueError("privileged skill waypoint action must be a 7D tuple")
        if any(not math.isfinite(value) for value in self.action):
            raise ValueError("privileged skill waypoint action must contain only finite values")
        for field_name in ("skill_index", "waypoint_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class PrivilegedSkillTrajectory:
    """Privileged skill trajectory that is explicitly ineligible for deployment."""

    seed: int
    initial_fingerprint: str
    deal: TexasHoldemDealIdentity
    selected_card_names: tuple[str, ...]
    waypoints: tuple[PrivilegedSkillWaypoint, ...]
    shadow_task_success: bool
    executor_label: str = SIMULATOR_PRIVILEGED_SKILL_EXECUTOR
    deployable: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("privileged skill trajectory seed must be an integer")
        if self.seed < 0:
            raise ValueError("privileged skill trajectory seed must be non-negative")
        if not isinstance(self.initial_fingerprint, str) or not self.initial_fingerprint.strip():
            raise ValueError("initial_fingerprint must be a non-empty string")
        if not isinstance(self.deal, TexasHoldemDealIdentity):
            raise TypeError("deal must be a TexasHoldemDealIdentity")
        selected_card_names = non_empty_unique_names(
            self.selected_card_names,
            field_name="selected_card_names",
        )
        missing = set(selected_card_names).difference(self.deal.card_names)
        if missing:
            raise ValueError(f"selected cards are missing from the deal: {sorted(missing)}")
        waypoints = tuple(self.waypoints)
        if not waypoints:
            raise ValueError("privileged skill trajectory must contain waypoints")
        if any(not isinstance(waypoint, PrivilegedSkillWaypoint) for waypoint in waypoints):
            raise TypeError("waypoints must contain PrivilegedSkillWaypoint values")
        if not isinstance(self.shadow_task_success, bool):
            raise TypeError("shadow_task_success must be a bool")
        if self.executor_label != SIMULATOR_PRIVILEGED_SKILL_EXECUTOR:
            raise ValueError(
                f"executor_label must remain {SIMULATOR_PRIVILEGED_SKILL_EXECUTOR!r} "
                "so privileged results cannot be mislabeled"
            )
        if self.deployable is not False:
            raise ValueError("a privileged simulator-skill trajectory cannot be deployable")
        object.__setattr__(self, "initial_fingerprint", self.initial_fingerprint.strip())
        object.__setattr__(self, "selected_card_names", selected_card_names)
        object.__setattr__(self, "waypoints", waypoints)

    @property
    def is_oracle_upper_bound(self) -> bool:
        return frozenset(self.selected_card_names) == frozenset(self.deal.target_card_names)

    @property
    def evaluation_label(self) -> str:
        if self.is_oracle_upper_bound:
            return SIMULATOR_ORACLE_UPPER_BOUND
        return SIMULATOR_PRIVILEGED_PLAN_CONTROL

    @property
    def oracle_label(self) -> str:
        """Backward-readable label; only contains ``oracle`` for the true plan."""

        return self.evaluation_label


@dataclass(frozen=True, slots=True)
class PrivilegedSkillReplay:
    """Compact result of endpoint-only replay on the main environment."""

    trajectory: PrivilegedSkillTrajectory
    waypoint_steps: int
    settle_steps: int
    success: bool
    terminated: bool
    truncated: bool
    final_outcome: EpisodeStep | None

    @property
    def total_steps(self) -> int:
        return self.waypoint_steps + self.settle_steps

    @property
    def evaluation_label(self) -> str:
        return self.trajectory.evaluation_label


def non_empty_unique_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    names = tuple(values)
    if not names:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = tuple(name.strip() for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def finite_non_negative(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return normalized


def validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")


# Stable compatibility names for records emitted before the executor/oracle
# distinction was made explicit.
PrivilegedOracleWaypoint = PrivilegedSkillWaypoint
PrivilegedOracleTrajectory = PrivilegedSkillTrajectory
PrivilegedOracleReplay = PrivilegedSkillReplay


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
]
