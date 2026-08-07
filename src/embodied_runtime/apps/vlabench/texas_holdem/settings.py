"""Validated values and settings for the matched Texas Hold'em experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..texas_holdem_task import TEXAS_HOLDEM_COMPOSITE_PROMPT, TEXAS_HOLDEM_TASK


@dataclass(frozen=True, slots=True)
class PokerCard:
    """One card exposed by the simulator oracle."""

    name: str
    value: str
    suit: str

    def __post_init__(self) -> None:
        for field_name in ("name", "value", "suit"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

    @property
    def primitive_prompt(self) -> str:
        return f"primitive: Please pick the poker {self.value} of {self.suit}"


@dataclass(frozen=True, slots=True)
class PokerDeal:
    """Immutable ground-truth deal captured before an episode starts."""

    cards: tuple[PokerCard, ...]
    target_names: tuple[str, ...]
    hand_type: str

    def __post_init__(self) -> None:
        cards = tuple(self.cards)
        target_names = tuple(self.target_names)
        if not cards:
            raise ValueError("Texas Hold'em deal must contain cards")
        if not target_names:
            raise ValueError("Texas Hold'em deal must contain at least one target")
        by_name = {card.name: card for card in cards}
        if len(by_name) != len(cards):
            raise ValueError("Texas Hold'em card names must be unique")
        missing = set(target_names).difference(by_name)
        if missing:
            raise ValueError(f"target cards are missing from the deal: {sorted(missing)}")
        if len(set(target_names)) != len(target_names):
            raise ValueError("target card names must be unique")
        if not isinstance(self.hand_type, str) or not self.hand_type.strip():
            raise ValueError("hand_type must be a non-empty string")
        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "target_names", target_names)
        object.__setattr__(self, "hand_type", self.hand_type.strip())

    @property
    def targets(self) -> tuple[PokerCard, ...]:
        by_name = {card.name: card for card in self.cards}
        return tuple(by_name[name] for name in self.target_names)


@dataclass(frozen=True, slots=True)
class EndpointPlaceControllerConfig:
    """Tunable Cartesian waypoint controller shared by all conditions."""

    lift_height_m: float = 0.15
    clearance_m: float = 0.12
    retract_height_m: float = 0.10
    max_translation_step_m: float = 0.025
    open_steps: int = 5
    slot_count: int = 5
    slot_spacing_m: float = 0.05
    workspace_abs_limit_m: float = 2.0

    def __post_init__(self) -> None:
        for field_name in (
            "lift_height_m",
            "clearance_m",
            "retract_height_m",
            "max_translation_step_m",
            "slot_spacing_m",
            "workspace_abs_limit_m",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and greater than zero")
            object.__setattr__(self, field_name, value)
        for field_name in ("open_steps", "slot_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class TexasHoldemExperimentConfig:
    checkpoint: str
    output_dir: Path
    seeds: tuple[int, ...] = (1000,)
    task: str = TEXAS_HOLDEM_TASK
    device: str = "cuda"
    backbone_path: str | None = None
    max_episode_steps: int = 800
    control_period_s: float = 0.1
    warmup_policy: bool = True
    local_files_only: bool = True
    place_controller: EndpointPlaceControllerConfig = EndpointPlaceControllerConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, str) or not self.checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty string")
        object.__setattr__(self, "checkpoint", self.checkpoint.strip())
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if (
            isinstance(self.seeds, (str, bytes))
            or not isinstance(self.seeds, Sequence)
            or not self.seeds
        ):
            raise ValueError("seeds must be a non-empty sequence")
        seeds = tuple(self.seeds)
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", seeds)
        if self.task != TEXAS_HOLDEM_TASK:
            raise ValueError("this experiment supports only the texas_holdem registry task")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.backbone_path is not None and not self.backbone_path.strip():
            raise ValueError("backbone_path must be non-empty when provided")
        if (
            isinstance(self.max_episode_steps, bool)
            or not isinstance(self.max_episode_steps, int)
            or self.max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        if (
            isinstance(self.control_period_s, bool)
            or not isinstance(self.control_period_s, (int, float))
            or not math.isfinite(float(self.control_period_s))
            or self.control_period_s <= 0
        ):
            raise ValueError("control_period_s must be finite and greater than zero")
        object.__setattr__(self, "control_period_s", float(self.control_period_s))
        if not isinstance(self.warmup_policy, bool):
            raise TypeError("warmup_policy must be a bool")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be a bool")
        if not isinstance(self.place_controller, EndpointPlaceControllerConfig):
            raise TypeError("place_controller must be an EndpointPlaceControllerConfig")


__all__ = [
    "TEXAS_HOLDEM_COMPOSITE_PROMPT",
    "TEXAS_HOLDEM_TASK",
    "EndpointPlaceControllerConfig",
    "PokerCard",
    "PokerDeal",
    "TexasHoldemExperimentConfig",
]
