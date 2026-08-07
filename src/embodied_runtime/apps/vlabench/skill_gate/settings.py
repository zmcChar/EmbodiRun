"""Validated values for the privileged-executor planner skill gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from embodied_runtime.integrations.planning.hf_texas_holdem import TexasHoldemCard


class PlannerSkillCondition(str, Enum):
    """Plan source compared under the same privileged low-level executor."""

    WRONG_PLAN = "wrong_plan"
    ORACLE_PLAN = "oracle_plan"
    CLOUD_PLAN = "cloud_plan"


@dataclass(frozen=True, slots=True)
class PlannerPokerDeal:
    """Structured card observation and simulator truth for one reset."""

    cards: tuple[TexasHoldemCard, ...]
    target_card_names: tuple[str, ...]
    hand_type: str

    def __post_init__(self) -> None:
        cards = tuple(self.cards)
        targets = tuple(self.target_card_names)
        if len(cards) < 5 or any(not isinstance(card, TexasHoldemCard) for card in cards):
            raise ValueError("a planning deal requires at least five TexasHoldemCard values")
        names = tuple(card.name for card in cards)
        if len(set(names)) != len(names):
            raise ValueError("planning deal card names must be unique")
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("planning deal targets must be non-empty and unique")
        missing = set(targets).difference(names)
        if missing:
            raise ValueError(f"planning targets are absent from the deal: {sorted(missing)}")
        if not isinstance(self.hand_type, str) or not self.hand_type.strip():
            raise ValueError("hand_type must be a non-empty string")
        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "target_card_names", targets)
        object.__setattr__(self, "hand_type", self.hand_type.strip())

    @property
    def card_names(self) -> tuple[str, ...]:
        return tuple(card.name for card in self.cards)


@dataclass(frozen=True, slots=True)
class PlannerSkillGateConfig:
    """Runtime settings for paired privileged-executor trials."""

    output_dir: Path
    seeds: tuple[int, ...] = (1000,)
    render_resolution: tuple[int, int] = (96, 96)
    max_episode_steps: int = 1200
    shadow_max_episode_steps: int = 2000
    settle_repeats: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        seeds = tuple(self.seeds)
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", seeds)
        resolution = tuple(self.render_resolution)
        if len(resolution) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        ):
            raise ValueError("render_resolution must contain two positive integers")
        object.__setattr__(self, "render_resolution", resolution)
        for field_name in ("max_episode_steps", "shadow_max_episode_steps"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.settle_repeats, bool)
            or not isinstance(self.settle_repeats, int)
            or self.settle_repeats < 0
        ):
            raise ValueError("settle_repeats must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PlannerSkillTrial:
    """One auditable condition result."""

    pair_id: str
    seed: int
    condition: PlannerSkillCondition
    initial_fingerprint: str
    hand_type: str
    true_target_card_names: tuple[str, ...]
    selected_card_names: tuple[str, ...]
    planner_accepted: bool
    planner_latency_ms: float
    execution_attempted: bool
    success: bool
    waypoint_steps: int
    settle_steps: int
    execution_label: str | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["condition"] = self.condition.value
        return document


__all__ = [
    "PlannerPokerDeal",
    "PlannerSkillCondition",
    "PlannerSkillGateConfig",
    "PlannerSkillTrial",
]
