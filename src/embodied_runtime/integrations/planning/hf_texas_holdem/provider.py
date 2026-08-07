"""Planning provider that turns grounded model output into task plans."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from embodied_runtime.tasks.planning import PlanEnvelope, PlanRequest, PlanStep

from .config import HfTexasHoldemPlannerConfig
from .errors import PlannerInputError, PlannerOutputError, PlannerRuntimeError
from .parsing import parse_texas_holdem_selection
from .prompt import build_texas_holdem_prompt
from .request_data import cards_from_request
from .runtime import TextGenerator, load_huggingface_generator


class HfTexasHoldemPlanner:
    """Generate grounded task plans from structured card metadata."""

    def __init__(
        self,
        config: HfTexasHoldemPlannerConfig,
        *,
        clock: Callable[[], float] = time.time,
        generator_factory: Callable[[], TextGenerator] | None = None,
    ) -> None:
        if not isinstance(config, HfTexasHoldemPlannerConfig):
            raise TypeError("config must be an HfTexasHoldemPlannerConfig")
        self.config = config
        self._clock = clock
        self._generator_factory = generator_factory
        self._generator: TextGenerator | None = None
        self._runtime_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._generator is not None

    def plan(self, request: PlanRequest) -> PlanEnvelope:
        if not isinstance(request, PlanRequest):
            raise TypeError("request must be a PlanRequest")
        cards = cards_from_request(request)
        if request.goal.allowed_skills and self.config.skill not in request.goal.allowed_skills:
            raise PlannerInputError(
                f"planner skill {self.config.skill!r} is not allowed by the task goal"
            )
        prompt = build_texas_holdem_prompt(
            instruction=request.goal.instruction,
            cards=cards,
        )
        with self._runtime_lock:
            generator = self._get_or_load_generator()
            try:
                response = generator.generate(prompt)
            except (PlannerInputError, PlannerOutputError, PlannerRuntimeError):
                raise
            except Exception as error:
                raise PlannerRuntimeError("planner text generation failed") from error
        selection = parse_texas_holdem_selection(
            response,
            cards=cards,
            allow_single_json_fence=self.config.allow_single_json_fence,
        )
        card_by_name = {card.name: card for card in cards}
        steps = tuple(
            PlanStep(
                step_id=f"select_card_{index:02d}",
                instruction=card_by_name[name].primitive_instruction,
                skill=self.config.skill,
                success_criteria=f"{name} is contained by the target placemat.",
                metadata={
                    "poker_name": name,
                    "target_index": index - 1,
                    "hand_type": selection.hand_type,
                },
            )
            for index, name in enumerate(selection.target_card_names, start=1)
        )
        created_at_s = _valid_clock_value(self._clock())
        revision = 1 if request.active_revision is None else request.active_revision + 1
        return PlanEnvelope(
            request_id=request.request_id,
            task_id=request.goal.task_id,
            session_id=request.goal.session_id,
            revision=revision,
            steps=steps,
            created_at_s=created_at_s,
            expires_at_s=created_at_s + self.config.plan_ttl_s,
            based_on_observation_id=request.observation_id,
            metadata={
                "planner": "huggingface_texas_holdem",
                "checkpoint": self.config.checkpoint,
                "hand_type": selection.hand_type,
                "target_card_names": selection.target_card_names,
            },
        )

    async def plan_async(self, request: PlanRequest) -> PlanEnvelope:
        return await asyncio.to_thread(self.plan, request)

    def _get_or_load_generator(self) -> TextGenerator:
        if self._generator is None:
            factory = self._generator_factory
            try:
                generator = (
                    factory() if factory is not None else load_huggingface_generator(self.config)
                )
            except PlannerRuntimeError:
                raise
            except Exception as error:
                raise PlannerRuntimeError(
                    f"failed to load planner checkpoint {self.config.checkpoint!r}"
                ) from error
            if not isinstance(generator, TextGenerator):
                raise PlannerRuntimeError("generator factory returned an incompatible object")
            self._generator = generator
        return self._generator


def _valid_clock_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerRuntimeError("planner clock must return a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise PlannerRuntimeError("planner clock must return a finite non-negative value")
    return normalized


__all__ = ["HfTexasHoldemPlanner"]
