"""Planner-quality gate with one fixed privileged VLABench skill executor.

This experiment is intentionally separate from the learned SmolVLA comparison.
It holds low-level execution constant with VLABench's simulator-privileged
expert and changes only the selected cards:

* ``wrong_plan`` replaces one true target with one distractor;
* ``oracle_plan`` selects the simulator's true best hand;
* ``cloud_plan`` optionally consumes a validated :class:`PlanEnvelope`.

The result is a subsystem gate, not a deployable robot result.  A positive
Oracle-vs-Wrong effect shows that plan content matters when execution is
competent.  It does not hide a weak learned low-level policy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from embodied_runtime.contracts import PlanEnvelope, PlanRequest, TaskGoal
from embodied_runtime.distributed import PlanManager
from embodied_runtime.integrations.planning import TexasHoldemCard
from embodied_runtime.simulators import (
    VLABenchSimulatorEndpoint,
    run_texas_holdem_privileged_skill_executor,
)

from .vlabench_texas_holdem import (
    TEXAS_HOLDEM_COMPOSITE_PROMPT,
    TEXAS_HOLDEM_TASK,
)


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
    conditions = [
        PlannerSkillCondition.WRONG_PLAN,
        PlannerSkillCondition.ORACLE_PLAN,
    ]
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
                fingerprint = _fingerprint(endpoint)
                deal = deal_inspector(endpoint)
                if expected_fingerprint is None:
                    expected_fingerprint = fingerprint
                    expected_deal = deal
                elif fingerprint != expected_fingerprint or deal != expected_deal:
                    raise RuntimeError(
                        f"seed {seed} did not reproduce the paired initial state and deal"
                    )
                trial = _run_condition(
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
                trials.append(trial)
        finally:
            endpoint.close()
        _write_gate_report(config, trials)
    return tuple(trials)


def _run_condition(
    *,
    config: PlannerSkillGateConfig,
    endpoint: Any,
    observation: Any,
    fingerprint: str,
    deal: PlannerPokerDeal,
    seed: int,
    condition: PlannerSkillCondition,
    cloud_planner: Any | None,
    endpoint_factory: Callable[..., Any],
    skill_executor: Callable[..., Any],
    clock: Callable[[], float],
) -> PlannerSkillTrial:
    pair_id = f"{TEXAS_HOLDEM_TASK}-seed-{seed:08d}"
    selected: tuple[str, ...] = ()
    planner_latency_ms = 0.0
    planner_accepted = True
    error: Exception | None = None

    if condition is PlannerSkillCondition.ORACLE_PLAN:
        selected = deal.target_card_names
    elif condition is PlannerSkillCondition.WRONG_PLAN:
        selected = deterministic_wrong_card_selection(deal, seed=seed)
    else:
        if cloud_planner is None:
            raise RuntimeError("cloud_plan condition requires a cloud planner")
        request = build_cloud_plan_request(
            deal=deal,
            seed=seed,
            fingerprint=fingerprint,
            observation_timestamp_s=observation.timestamp_s,
        )
        started = _clock_value(clock)
        try:
            envelope = _call_cloud_planner(cloud_planner, request)
            planner_latency_ms = (_clock_value(clock) - started) * 1000.0
            selected = _activate_and_extract_cloud_selection(envelope, request)
        except Exception as caught:  # noqa: BLE001 - model/transport errors are result data
            planner_latency_ms = max(0.0, (_clock_value(clock) - started) * 1000.0)
            planner_accepted = False
            error = caught

    if not planner_accepted:
        return PlannerSkillTrial(
            pair_id=pair_id,
            seed=seed,
            condition=condition,
            initial_fingerprint=fingerprint,
            hand_type=deal.hand_type,
            true_target_card_names=deal.target_card_names,
            selected_card_names=selected,
            planner_accepted=False,
            planner_latency_ms=planner_latency_ms,
            execution_attempted=False,
            success=False,
            waypoint_steps=0,
            settle_steps=0,
            execution_label=None,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

    def shadow_factory(task: str, *, max_episode_steps: int) -> Any:
        return endpoint_factory(
            task,
            max_episode_steps=max_episode_steps,
            render_resolution=config.render_resolution,
        )

    try:
        replay = skill_executor(
            endpoint,
            seed=seed,
            selected_card_names=selected,
            settle_repeats=config.settle_repeats,
            shadow_endpoint_factory=shadow_factory,
            shadow_max_episode_steps=config.shadow_max_episode_steps,
        )
    except Exception as caught:  # noqa: BLE001 - simulator failures are result data
        return PlannerSkillTrial(
            pair_id=pair_id,
            seed=seed,
            condition=condition,
            initial_fingerprint=fingerprint,
            hand_type=deal.hand_type,
            true_target_card_names=deal.target_card_names,
            selected_card_names=selected,
            planner_accepted=True,
            planner_latency_ms=planner_latency_ms,
            execution_attempted=True,
            success=False,
            waypoint_steps=0,
            settle_steps=0,
            execution_label=None,
            error_type=type(caught).__name__,
            error_message=str(caught),
        )
    return PlannerSkillTrial(
        pair_id=pair_id,
        seed=seed,
        condition=condition,
        initial_fingerprint=fingerprint,
        hand_type=deal.hand_type,
        true_target_card_names=deal.target_card_names,
        selected_card_names=selected,
        planner_accepted=True,
        planner_latency_ms=planner_latency_ms,
        execution_attempted=True,
        success=bool(replay.success),
        waypoint_steps=int(replay.waypoint_steps),
        settle_steps=int(replay.settle_steps),
        execution_label=str(replay.trajectory.evaluation_label),
    )


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
    omitted = min(
        deal.target_card_names,
        key=lambda name: _stable_rank(seed, "omit", name),
    )
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


def _call_cloud_planner(planner: Any, request: PlanRequest) -> PlanEnvelope:
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


def _activate_and_extract_cloud_selection(
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


def _fingerprint(endpoint: Any) -> str:
    value = getattr(endpoint, "initial_fingerprint", None)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("endpoint must expose a fingerprint after reset")
    return value.strip()


def _stable_rank(seed: int, purpose: str, card_name: str) -> bytes:
    return hashlib.sha256(
        f"vlabench-planner-skill-gate-v1:{seed}:{purpose}:{card_name}".encode()
    ).digest()


def _clock_value(clock: Callable[[], float]) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("clock must return a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("clock must return a finite value")
    return normalized


def _comparison(
    trials: Sequence[PlannerSkillTrial],
    *,
    baseline: PlannerSkillCondition,
    comparator: PlannerSkillCondition,
) -> dict[str, Any] | None:
    baseline_by_pair = {trial.pair_id: trial for trial in trials if trial.condition is baseline}
    comparator_by_pair = {trial.pair_id: trial for trial in trials if trial.condition is comparator}
    pairs = sorted(set(baseline_by_pair) & set(comparator_by_pair))
    if not pairs:
        return None
    baseline_only = 0
    comparator_only = 0
    both_success = 0
    both_fail = 0
    for pair_id in pairs:
        left = baseline_by_pair[pair_id].success
        right = comparator_by_pair[pair_id].success
        if left and right:
            both_success += 1
        elif left:
            baseline_only += 1
        elif right:
            comparator_only += 1
        else:
            both_fail += 1
    return {
        "baseline": baseline.value,
        "comparator": comparator.value,
        "matched_pairs": len(pairs),
        "both_succeeded": both_success,
        "both_failed": both_fail,
        "baseline_only_succeeded": baseline_only,
        "comparator_only_succeeded": comparator_only,
        "effect_percentage_points": (comparator_only - baseline_only) / len(pairs) * 100.0,
    }


def _write_gate_report(
    config: PlannerSkillGateConfig,
    trials: Sequence[PlannerSkillTrial],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = [
        comparison
        for comparison in (
            _comparison(
                trials,
                baseline=PlannerSkillCondition.WRONG_PLAN,
                comparator=PlannerSkillCondition.ORACLE_PLAN,
            ),
            _comparison(
                trials,
                baseline=PlannerSkillCondition.WRONG_PLAN,
                comparator=PlannerSkillCondition.CLOUD_PLAN,
            ),
        )
        if comparison is not None
    ]
    document = {
        "schema_version": 1,
        "claim_scope": (
            "planner-quality subsystem gate with a non-deployable simulator-privileged "
            "skill executor; not an end-to-end learned-policy result"
        ),
        "config": {
            "seeds": list(config.seeds),
            "render_resolution": list(config.render_resolution),
            "max_episode_steps": config.max_episode_steps,
            "shadow_max_episode_steps": config.shadow_max_episode_steps,
            "settle_repeats": config.settle_repeats,
        },
        "trials": [trial.to_dict() for trial in trials],
        "comparisons": comparisons,
    }
    (config.output_dir / "planner_skill_gate.json").write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the VLABench Oracle/Wrong/Cloud plan skill-executor gate."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(1000,))
    parser.add_argument("--render-size", type=int, default=96)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--shadow-max-episode-steps", type=int, default=2000)
    parser.add_argument("--settle-repeats", type=int, default=12)
    parser.add_argument("--planner-checkpoint")
    parser.add_argument("--planner-device", default="cpu")
    parser.add_argument("--planner-dtype", default="float32")
    parser.add_argument("--planner-max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--allow-single-json-fence",
        action="store_true",
        help="accept exactly one bare ```json ... ``` wrapper before schema validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    planner = None
    if args.planner_checkpoint:
        from embodied_runtime.integrations.planning import (
            HfTexasHoldemPlanner,
            HfTexasHoldemPlannerConfig,
        )

        planner = HfTexasHoldemPlanner(
            HfTexasHoldemPlannerConfig(
                checkpoint=args.planner_checkpoint,
                device=args.planner_device,
                dtype=args.planner_dtype,
                max_new_tokens=args.planner_max_new_tokens,
                allow_single_json_fence=args.allow_single_json_fence,
            )
        )
    config = PlannerSkillGateConfig(
        output_dir=args.output_dir,
        seeds=args.seeds,
        render_resolution=(args.render_size, args.render_size),
        max_episode_steps=args.max_episode_steps,
        shadow_max_episode_steps=args.shadow_max_episode_steps,
        settle_repeats=args.settle_repeats,
    )
    trials = run_planner_skill_gate(config, cloud_planner=planner)
    print(json.dumps([trial.to_dict() for trial in trials], allow_nan=False, indent=2))
    return 0


__all__ = [
    "PlannerPokerDeal",
    "PlannerSkillCondition",
    "PlannerSkillGateConfig",
    "PlannerSkillTrial",
    "build_cloud_plan_request",
    "build_parser",
    "deterministic_wrong_card_selection",
    "inspect_planner_poker_deal",
    "main",
    "run_planner_skill_gate",
]
