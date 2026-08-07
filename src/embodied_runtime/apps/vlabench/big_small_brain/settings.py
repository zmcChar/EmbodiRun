"""Validated settings for the paired get-coffee planning pilot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from embodied_runtime.tasks.planning import PlanStep

GET_COFFEE_TASK = "get_coffee"
GET_COFFEE_COMPOSITE_PROMPT = "composite: Get me a cup of coffee."
GET_COFFEE_ORACLE_STEPS = (
    PlanStep(
        step_id="place_mug",
        skill="place_mug",
        instruction="primitive: Pick up the mug and place it under the coffee dispenser.",
        success_criteria="The mug is contained by the coffee machine.",
        metadata={"step_budget": 350},
    ),
    PlanStep(
        step_id="press_start",
        skill="press_start",
        instruction="primitive: Press the start button on the coffee machine.",
        success_criteria="The coffee machine is active while the mug is under it.",
        metadata={"step_budget": 150},
    ),
)


@dataclass(frozen=True, slots=True)
class VLABenchPilotConfig:
    checkpoint: str
    output_dir: Path
    seeds: tuple[int, ...] = (1000,)
    task: str = GET_COFFEE_TASK
    device: str = "cuda"
    backbone_path: str | None = None
    max_episode_steps: int = 500
    first_subgoal_budget: int = 350
    subgoal_stability_steps: int = 3
    control_period_s: float = 0.1
    warmup_policy: bool = True
    local_files_only: bool = True

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
        if self.task != GET_COFFEE_TASK:
            raise ValueError("the first paired pilot currently supports only get_coffee")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.backbone_path is not None and not self.backbone_path.strip():
            raise ValueError("backbone_path must be non-empty when provided")
        for name in ("max_episode_steps", "first_subgoal_budget", "subgoal_stability_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.first_subgoal_budget >= self.max_episode_steps:
            raise ValueError("first_subgoal_budget must be smaller than max_episode_steps")
        if (
            isinstance(self.control_period_s, bool)
            or not isinstance(self.control_period_s, (int, float))
            or self.control_period_s <= 0
        ):
            raise ValueError("control_period_s must be greater than zero")
        if not isinstance(self.warmup_policy, bool):
            raise TypeError("warmup_policy must be a bool")


__all__ = [
    "GET_COFFEE_COMPOSITE_PROMPT",
    "GET_COFFEE_ORACLE_STEPS",
    "GET_COFFEE_TASK",
    "VLABenchPilotConfig",
]
