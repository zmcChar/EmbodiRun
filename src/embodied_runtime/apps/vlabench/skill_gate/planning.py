"""Deal inspection and plan-source adaptation for the planner skill gate."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from embodied_runtime.distributed import PlanManager
from embodied_runtime.integrations.planning.hf_texas_holdem import TexasHoldemCard
from embodied_runtime.tasks.planning import PlanEnvelope, PlanRequest, TaskGoal

from ..texas_holdem_task import TEXAS_HOLDEM_COMPOSITE_PROMPT, TEXAS_HOLDEM_TASK
from .settings import PlannerPokerDeal


def inspect_planner_poker_deal(endpoint: Any) -> PlannerPokerDeal:
    """Read structured cards from a live simulator adapter."""

    raw = endpoint.raw_environment
    task = getattr(getattr(raw, "_env", None), "task", None)
    if task is None:
        raise RuntimeError("VLABench endpoint did not expose a live task")
    cards = tuple(
        TexasHoldemCard(
            name=str(getattr(card, "name", "")),
            value=str(getattr(card, "value", "")),
            suit=str(getattr(card, "suite", "")),
        )
        for card in tuple(getattr(task, "pokers", ()))
    )
    targets = getattr(task, "target_entities", None)
    if isinstance(targets, Mapping) or (
        isinstance(targets, Sequence) and not isinstance(targets, (str, bytes))
    ):
        target_names = tuple(str(name) for name in targets)
    else:
        raise TypeError("Texas Hold'em task did not expose target card names")
    return PlannerPokerDeal(
        cards=cards,
        target_card_names=target_names,
        hand_type=str(getattr(task, "max_cardtype", "")),
    )


def deterministic_wrong_card_selection(
    deal: PlannerPokerDeal,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Replace exactly one target with a deterministic distractor."""

    if not isinstance(deal, PlannerPokerDeal):
        raise TypeError("deal must be a PlannerPokerDeal")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    target_set = set(deal.target_card_names)
    distractors = tuple(name for name in deal.card_names if name not in target_set)
    if not distractors:
        raise RuntimeError("a wrong-plan control requires at least one distractor")
    omitted = min(deal.target_card_names, key=lambda name: _stable_rank(seed, "omit", name))
    replacement = min(distractors, key=lambda name: _stable_rank(seed, "replace", name))
    selection = tuple(name for name in deal.target_card_names if name != omitted) + (replacement,)
    return tuple(sorted(selection, key=lambda name: _stable_rank(seed, "order", name)))


def build_cloud_plan_request(
    *,
    deal: PlannerPokerDeal,
    seed: int,
    fingerprint: str,
    observation_timestamp_s: float,
) -> PlanRequest:
    """Publish structured observations without leaking target identities."""

    cards = [{"name": card.name, "value": card.value, "suit": card.suit} for card in deal.cards]
    goal = TaskGoal(
        task_id=TEXAS_HOLDEM_TASK,
        session_id=f"texas-cloud-{seed}",
        instruction=TEXAS_HOLDEM_COMPOSITE_PROMPT,
        allowed_skills=("pick_and_place_poker",),
        metadata={"texas_holdem": {"cards": cards}, "seed": seed},
    )
    return PlanRequest(
        goal=goal,
        observation_id=fingerprint,
        observation_timestamp_s=observation_timestamp_s,
    )


def call_cloud_planner(planner: Any, request: PlanRequest) -> PlanEnvelope:
    plan_async = getattr(planner, "plan_async", None)
    if callable(plan_async):
        result = asyncio.run(plan_async(request))
    else:
        plan = getattr(planner, "plan", None)
        if not callable(plan):
            raise TypeError("cloud planner must expose plan_async(request) or plan(request)")
        result = plan(request)
    if not isinstance(result, PlanEnvelope):
        raise TypeError("cloud planner did not return a PlanEnvelope")
    return result


def activate_and_extract_cloud_selection(
    envelope: PlanEnvelope,
    request: PlanRequest,
) -> tuple[str, ...]:
    manager = PlanManager(request.goal.session_id)
    manager.offer_plan(envelope, request=request)
    activated = manager.activate_pending(safe_boundary=True)
    if activated is None:
        raise RuntimeError("validated cloud plan could not be activated at reset")
    selected: list[str] = []
    for step in activated.steps:
        name = step.metadata.get("poker_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"cloud plan step {step.step_id!r} has no grounded poker_name")
        selected.append(name.strip())
    if len(set(selected)) != len(selected):
        raise ValueError("cloud plan selected duplicate cards")
    return tuple(selected)


def endpoint_fingerprint(endpoint: Any) -> str:
    value = getattr(endpoint, "initial_fingerprint", None)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("endpoint must expose a fingerprint after reset")
    return value.strip()


def clock_value(clock: Callable[[], float]) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("clock must return a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("clock must return a finite value")
    return normalized


def _stable_rank(seed: int, purpose: str, card_name: str) -> bytes:
    return hashlib.sha256(
        f"vlabench-planner-skill-gate-v1:{seed}:{purpose}:{card_name}".encode()
    ).digest()


__all__ = ["build_cloud_plan_request", "deterministic_wrong_card_selection"]
