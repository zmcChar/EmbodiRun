"""Oracle and deterministic-control plan construction."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Any

from embodied_runtime.distributed import PlanManager
from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.tasks.planning import (
    PlanEnvelope,
    PlanRequest,
    PlanStep,
    PlanStepStatus,
    TaskGoal,
)

from .settings import (
    TEXAS_HOLDEM_COMPOSITE_PROMPT,
    TEXAS_HOLDEM_TASK,
    PokerCard,
    PokerDeal,
)


def wrong_plan_cards(deal: PokerDeal, *, seed: int) -> tuple[PokerCard, ...]:
    """Replace exactly one target with a reproducible distractor."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    target_names = set(deal.target_names)
    distractors = tuple(card for card in deal.cards if card.name not in target_names)
    if not distractors:
        raise RuntimeError("cannot construct a wrong plan without a distractor card")
    omitted = min(deal.targets, key=lambda card: _control_rank(seed, "omit", card.name))
    replacement = min(
        distractors,
        key=lambda card: _control_rank(seed, "replace", card.name),
    )
    selected = tuple(card for card in deal.targets if card.name != omitted.name) + (replacement,)
    return tuple(sorted(selected, key=lambda card: _control_rank(seed, "order", card.name)))


def activate_card_plan(
    *,
    observation: RobotObservation,
    fingerprint: str,
    seed: int,
    deal: PokerDeal,
    cards: tuple[PokerCard, ...],
    planner_name: str,
) -> PlanManager:
    if not cards:
        raise ValueError("cards must not be empty")
    if not isinstance(planner_name, str) or not planner_name.strip():
        raise ValueError("planner_name must be a non-empty string")
    planner_name = planner_name.strip()
    session_id = f"texas-holdem-{planner_name}-{seed}"
    steps = tuple(
        PlanStep(
            step_id=f"place_{index}_{card.name}",
            skill="pick_and_place_poker",
            instruction=card.primitive_prompt,
            success_criteria=f"{card.name} is contained by the placemat.",
            metadata={
                "poker_name": card.name,
                "plan_index": index,
                "is_task_target": card.name in deal.target_names,
            },
        )
        for index, card in enumerate(cards)
    )
    goal = TaskGoal(
        task_id=TEXAS_HOLDEM_TASK,
        session_id=session_id,
        instruction=TEXAS_HOLDEM_COMPOSITE_PROMPT,
        allowed_skills=("pick_and_place_poker",),
        metadata={"planner": planner_name, "seed": seed, "hand_type": deal.hand_type},
    )
    request = PlanRequest(
        goal=goal,
        observation=observation.values,
        observation_id=fingerprint,
        observation_timestamp_s=observation.timestamp_s,
    )
    created_at_s = time.time()
    envelope = PlanEnvelope(
        request_id=request.request_id,
        task_id=goal.task_id,
        session_id=goal.session_id,
        revision=1,
        steps=steps,
        created_at_s=created_at_s,
        expires_at_s=created_at_s + 3600.0,
        based_on_observation_id=fingerprint,
        metadata={"planner": planner_name, "hand_type": deal.hand_type},
    )
    manager = PlanManager(session_id)
    manager.offer_plan(envelope, request=request)
    if manager.activate_pending(safe_boundary=True) is None:
        raise RuntimeError(f"{planner_name} plan could not be activated at episode reset")
    return manager


def advance_completed_plan_steps(
    manager: PlanManager,
    completed_cards: frozenset[str],
) -> None:
    while manager.current_step is not None:
        poker_name = manager.current_step.metadata.get("poker_name")
        if poker_name not in completed_cards:
            return
        manager.advance(PlanStepStatus.SUCCEEDED)
        if manager.current_step is not None:
            manager.record_feedback(PlanStepStatus.ACTIVE)


def deal_document(deal: PokerDeal) -> dict[str, Any]:
    return {
        "cards": [asdict(card) for card in deal.cards],
        "target_names": list(deal.target_names),
        "target_prompts": [card.primitive_prompt for card in deal.targets],
        "hand_type": deal.hand_type,
    }


def _control_rank(seed: int, purpose: str, card_name: str) -> bytes:
    material = f"vlabench-texas-wrong-plan-v1:{seed}:{purpose}:{card_name}"
    return hashlib.sha256(material.encode("utf-8")).digest()


__all__ = ["wrong_plan_cards"]
