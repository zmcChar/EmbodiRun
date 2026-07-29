"""Matched VLABench Texas Hold'em experiment for task-level reasoning.

This experiment deliberately separates two responsibilities:

* SmolVLA performs the same low-level card manipulation in every condition.
* The ``oracle_plan`` condition receives exact, training-distribution primitive
  prompts for the cards in the best Texas Hold'em combination.
* The ``wrong_plan`` negative control takes the identical plan and skill path,
  but deterministically replaces exactly one target with a distractor card.
* The ``edge_only`` condition receives the exact composite task prompt.

After any condition grasps a card, the same deterministic endpoint-level
controller carries it to the placemat.  This keeps the comparison focused on
card selection/reasoning instead of accidentally comparing two place policies.
Every simulator transition, including the deterministic place trajectory, goes
through :class:`~embodied_runtime.simulators.SimulatorEndpoint`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embodied_runtime.contracts import (
    PlanEnvelope,
    PlanRequest,
    PlanStep,
    PlanStepStatus,
    RobotAction,
    RobotObservation,
    TaskGoal,
)
from embodied_runtime.distributed import PlanManager
from embodied_runtime.evaluation import (
    EvaluationReport,
    ExperimentCondition,
    OracleGateConfig,
    TrialRecord,
    paired_binary_comparison,
)
from embodied_runtime.simulators import EpisodeStep, VLABenchSimulatorEndpoint

TEXAS_HOLDEM_TASK = "texas_holdem"
TEXAS_HOLDEM_COMPOSITE_PROMPT = (
    "composite: We're playing Texas hodl'em game! "
    "What's your largest cards? Show me on the placemat"
)
_TARGET_CONTAINER_NAME = "target_container"


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
    """Tunable Cartesian waypoint controller shared by both conditions.

    The defaults follow VLABench's own placement geometry: the placemat's
    ``get_place_point`` plus the Franka ``ee_offset``.  They are intentionally
    exposed because the first real pilot still needs to validate clearance and
    release height against the downloaded asset variants.
    """

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


@dataclass(frozen=True, slots=True)
class _EpisodeRun:
    record: TrialRecord
    deal: PokerDeal
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ControllerResult:
    observation: RobotObservation
    outcome: EpisodeStep | None
    steps: int
    events: tuple[dict[str, Any], ...]


class EndpointPlaceController:
    """Move a currently grasped card to the placemat through endpoint actions."""

    def __init__(self, config: EndpointPlaceControllerConfig) -> None:
        if not isinstance(config, EndpointPlaceControllerConfig):
            raise TypeError("config must be an EndpointPlaceControllerConfig")
        self.config = config

    def execute(
        self,
        *,
        endpoint: Any,
        observation: RobotObservation,
        available_steps: int,
        condition: ExperimentCondition,
        prompt: str,
        grasped_poker: str,
        placement_index: int,
    ) -> _ControllerResult:
        if not isinstance(observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        if isinstance(available_steps, bool) or not isinstance(available_steps, int):
            raise TypeError("available_steps must be an integer")
        if available_steps < 0:
            raise ValueError("available_steps must be non-negative")
        if placement_index < 0:
            raise ValueError("placement_index must be non-negative")
        if available_steps == 0:
            return _ControllerResult(observation, None, 0, ())

        target = _place_target_robot_frame(
            endpoint,
            placement_index=placement_index,
            config=self.config,
        )
        state = _agent_state(observation)
        waypoints = _build_place_waypoints(
            state,
            target,
            config=self.config,
        )
        events: list[dict[str, Any]] = []
        outcome: EpisodeStep | None = None
        current_observation = observation
        for waypoint_index, (phase, action_values) in enumerate(waypoints[:available_steps]):
            outcome = endpoint.step(
                RobotAction(
                    timestamp_s=time.time(),
                    values={"action": action_values},
                    metadata={
                        "condition": condition.value,
                        "prompt": prompt,
                        "controller": "shared_endpoint_place",
                        "phase": phase,
                        "grasped_poker": grasped_poker,
                        "placement_index": placement_index,
                        "waypoint_index": waypoint_index,
                    },
                )
            )
            current_observation = outcome.observation
            events.append(
                {
                    "event": "place_controller_step",
                    "phase": phase,
                    "grasped_poker": grasped_poker,
                    "placement_index": placement_index,
                    "waypoint_index": waypoint_index,
                    "success": bool(outcome.success),
                    "physics_error": bool(outcome.info.get("physics_error", False)),
                }
            )
            if outcome.done:
                break
        return _ControllerResult(
            observation=current_observation,
            outcome=outcome,
            steps=len(events),
            events=tuple(events),
        )


def _rotated_conditions(pair_index: int) -> tuple[ExperimentCondition, ...]:
    """Counterbalance execution order without changing a seed's conditions."""

    conditions = (
        ExperimentCondition.EDGE_ONLY,
        ExperimentCondition.ORACLE_PLAN,
        ExperimentCondition.WRONG_PLAN,
    )
    offset = pair_index % len(conditions)
    return conditions[offset:] + conditions[:offset]


def run_texas_holdem_experiment(
    config: TexasHoldemExperimentConfig,
    *,
    runner_factory: Callable[..., Any] | None = None,
    endpoint_factory: Callable[..., Any] | None = None,
    policy_seeder: Callable[[int], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    """Run matched Edge/Oracle/Wrong-plan trials and persist each completed seed."""

    if not isinstance(config, TexasHoldemExperimentConfig):
        raise TypeError("config must be a TexasHoldemExperimentConfig")
    runner_factory = runner_factory or _make_runner
    endpoint_factory = endpoint_factory or VLABenchSimulatorEndpoint
    policy_seeder = policy_seeder or _seed_policy

    runner = runner_factory(
        config.checkpoint,
        task=config.task,
        device=config.device,
        backbone_path=config.backbone_path,
        local_files_only=config.local_files_only,
    )
    runner.load()
    place_controller = EndpointPlaceController(config.place_controller)

    records: list[TrialRecord] = []
    trace_documents: list[dict[str, Any]] = []
    for pair_index, seed in enumerate(config.seeds):
        endpoint = endpoint_factory(
            config.task,
            max_episode_steps=config.max_episode_steps,
        )
        conditions = _rotated_conditions(pair_index)
        pair_runs: list[_EpisodeRun] = []
        try:
            if pair_index == 0 and config.warmup_policy:
                warmup_observation = endpoint.reset(seed=seed)
                policy_seeder(seed)
                runner.reset_action_queue()
                runner.select_action(
                    warmup_observation.values,
                    TEXAS_HOLDEM_COMPOSITE_PROMPT,
                )
                runner.reset_action_queue()
            for condition in conditions:
                pair_runs.append(
                    _run_episode(
                        config=config,
                        endpoint=endpoint,
                        runner=runner,
                        place_controller=place_controller,
                        condition=condition,
                        seed=seed,
                        policy_seeder=policy_seeder,
                        clock=clock,
                    )
                )
        finally:
            endpoint.close()

        fingerprints = tuple(run.record.initial_fingerprint for run in pair_runs)
        if len(fingerprints) != len(conditions) or any(value is None for value in fingerprints):
            raise RuntimeError(f"matched seed {seed} is missing an initial fingerprint")
        if len(set(fingerprints)) != 1:
            raise RuntimeError(
                f"matched seed {seed} did not reproduce the same initial observation"
            )
        if len({run.deal for run in pair_runs}) != 1:
            raise RuntimeError(f"matched seed {seed} did not reproduce the same poker deal")

        for run in pair_runs:
            records.append(run.record)
            trace_documents.append(
                {
                    "pair_id": run.record.pair_id,
                    "task_id": run.record.task_id,
                    "seed": run.record.seed,
                    "condition": run.record.condition.value,
                    "initial_fingerprint": run.record.initial_fingerprint,
                    "deal": _deal_document(run.deal),
                    "events": list(run.events),
                }
            )
        _write_outputs(
            config=config,
            report=EvaluationReport(records),
            traces=trace_documents,
            runtime_facts=getattr(runner, "facts", None),
        )

    return EvaluationReport(records)


def _run_episode(
    *,
    config: TexasHoldemExperimentConfig,
    endpoint: Any,
    runner: Any,
    place_controller: EndpointPlaceController,
    condition: ExperimentCondition,
    seed: int,
    policy_seeder: Callable[[int], None],
    clock: Callable[[], float],
) -> _EpisodeRun:
    observation: RobotObservation = endpoint.reset(seed=seed)
    fingerprint = endpoint.initial_fingerprint
    deal = _inspect_poker_deal(endpoint)
    policy_seeder(seed)
    runner.reset_action_queue()

    plan_manager: PlanManager | None = None
    planned_cards: tuple[PokerCard, ...] = ()
    planner_name: str | None = None
    planner_latencies_ms: list[float] = []
    if condition in {
        ExperimentCondition.ORACLE_PLAN,
        ExperimentCondition.WRONG_PLAN,
    }:
        planner_start_s = clock()
        if condition is ExperimentCondition.ORACLE_PLAN:
            planned_cards = deal.targets
            planner_name = "simulator_oracle"
        else:
            planned_cards = _wrong_plan_cards(deal, seed=seed)
            planner_name = "deterministic_wrong_control"
        plan_manager = _activate_card_plan(
            observation=observation,
            fingerprint=fingerprint,
            seed=seed,
            deal=deal,
            cards=planned_cards,
            planner_name=planner_name,
        )
        planner_latency_s = clock() - planner_start_s
        if planner_latency_s < 0:
            raise RuntimeError("clock moved backwards while measuring oracle planning")
        planner_latencies_ms.append(planner_latency_s * 1000.0)
        plan_manager.record_feedback(PlanStepStatus.ACTIVE)

    pair_id = f"{config.task}-seed-{seed:08d}"
    edge_latencies_ms: list[float] = []
    chunk_generation_latencies_ms: list[float] = []
    queue_hit_latencies_ms: list[float] = []
    deadline_misses = 0
    events: list[dict[str, Any]] = [
        {
            "event": "deal",
            **_deal_document(deal),
        }
    ]
    if plan_manager is not None:
        events.append(
            {
                "event": "plan_activated",
                "planner": planner_name,
                "planned_card_names": [card.name for card in planned_cards],
                "planned_prompts": [card.primitive_prompt for card in planned_cards],
                "deliberately_incorrect": (condition is ExperimentCondition.WRONG_PLAN),
            }
        )
    completed_targets: frozenset[str] = frozenset()
    completed_plan_cards: frozenset[str] = frozenset()
    placement_index = 0
    success = False
    steps = 0

    while steps < config.max_episode_steps:
        if condition is ExperimentCondition.EDGE_ONLY:
            prompt = TEXAS_HOLDEM_COMPOSITE_PROMPT
            plan_step_id = None
        else:
            assert plan_manager is not None
            _advance_completed_plan_steps(plan_manager, completed_plan_cards)
            current_step = plan_manager.current_step
            if current_step is None:
                break
            prompt = current_step.instruction
            plan_step_id = current_step.step_id

        start_s = clock()
        selected = runner.select_action(observation.values, prompt)
        full_latency_s = clock() - start_s
        if full_latency_s < 0:
            raise RuntimeError("clock moved backwards while measuring edge inference")
        latency_ms = full_latency_s * 1000.0
        edge_latencies_ms.append(latency_ms)
        model_latency_s = float(selected.inference_latency_s)
        if not math.isfinite(model_latency_s) or model_latency_s < 0:
            raise RuntimeError("runner returned an invalid inference latency")
        model_latency_ms = model_latency_s * 1000.0
        generated_chunk = bool(getattr(selected, "generated_chunk", False))
        if generated_chunk:
            chunk_generation_latencies_ms.append(model_latency_ms)
        else:
            queue_hit_latencies_ms.append(model_latency_ms)
        if full_latency_s > config.control_period_s:
            deadline_misses += 1

        outcome = endpoint.step(
            RobotAction(
                timestamp_s=time.time(),
                values={"action": selected.action},
                metadata={
                    "condition": condition.value,
                    "prompt": prompt,
                    "plan_step_id": plan_step_id,
                    "controller": "edge_policy",
                },
            )
        )
        steps += 1
        observation = outcome.observation
        events.append(
            {
                "event": "edge_policy_step",
                "step": steps,
                "prompt": prompt,
                "plan_step_id": plan_step_id,
                "edge_latency_ms": latency_ms,
                "model_latency_ms": model_latency_ms,
                "generated_chunk": generated_chunk,
                "queue_remaining": getattr(selected, "queue_remaining", None),
                "success": bool(outcome.success),
                "physics_error": bool(outcome.info.get("physics_error", False)),
            }
        )
        if outcome.success:
            success = True
            completed_targets = frozenset(deal.target_names)
            break
        if outcome.done:
            break

        grasped_poker = _grasped_poker_name(endpoint)
        if grasped_poker is not None:
            controller_result = place_controller.execute(
                endpoint=endpoint,
                observation=observation,
                available_steps=config.max_episode_steps - steps,
                condition=condition,
                prompt=prompt,
                grasped_poker=grasped_poker,
                placement_index=placement_index,
            )
            placement_index += 1
            steps += controller_result.steps
            observation = controller_result.observation
            first_controller_step = steps - controller_result.steps + 1
            for offset, event in enumerate(controller_result.events):
                events.append({"step": first_controller_step + offset, **event})
            # Cached policy actions were generated before the deterministic
            # controller changed both robot pose and scene state.
            runner.reset_action_queue()
            if controller_result.outcome is not None:
                if controller_result.outcome.success:
                    success = True
                    completed_targets = frozenset(deal.target_names)
                    break
                if controller_result.outcome.done:
                    break

        completed_targets = _contained_target_names(endpoint, deal)
        if plan_manager is not None:
            planned_names = tuple(card.name for card in planned_cards)
            completed_plan_cards = _contained_card_names(endpoint, planned_names)
            _advance_completed_plan_steps(plan_manager, completed_plan_cards)

    record = TrialRecord(
        pair_id=pair_id,
        task_id=config.task,
        seed=seed,
        condition=condition,
        success=success,
        steps=steps,
        edge_latencies_ms=tuple(edge_latencies_ms),
        planner_latencies_ms=tuple(planner_latencies_ms),
        chunk_generation_latencies_ms=tuple(chunk_generation_latencies_ms),
        queue_hit_latencies_ms=tuple(queue_hit_latencies_ms),
        deadline_misses=deadline_misses,
        subgoals_completed=len(completed_targets),
        subgoals_total=len(deal.target_names),
        initial_fingerprint=fingerprint,
    )
    return _EpisodeRun(record=record, deal=deal, events=tuple(events))


def _inspect_poker_deal(endpoint: Any) -> PokerDeal:
    task, _ = _task_and_physics(endpoint)
    pokers = tuple(getattr(task, "pokers", ()))
    if not pokers:
        raise RuntimeError("texas_holdem task did not expose task.pokers")
    cards = tuple(
        PokerCard(
            name=str(getattr(poker, "name", "")),
            value=str(getattr(poker, "value", "")),
            suit=str(getattr(poker, "suite", "")),
        )
        for poker in pokers
    )
    raw_targets = getattr(task, "target_entities", None)
    if isinstance(raw_targets, Mapping) or (
        isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes))
    ):
        target_names = tuple(str(name) for name in raw_targets)
    else:
        raise TypeError("texas_holdem task did not expose target_entities as a sequence")
    return PokerDeal(
        cards=cards,
        target_names=target_names,
        hand_type=str(getattr(task, "max_cardtype", "")),
    )


def _wrong_plan_cards(deal: PokerDeal, *, seed: int) -> tuple[PokerCard, ...]:
    """Create a reproducible hard negative with one target replaced.

    The control intentionally uses simulator labels so that it is guaranteed to
    be wrong.  It is not a candidate planner: its purpose is to detect task,
    executor, or metric leakage under a plan that has the same number and type
    of primitive skill calls as the oracle plan.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    target_names = set(deal.target_names)
    distractors = tuple(card for card in deal.cards if card.name not in target_names)
    if not distractors:
        raise RuntimeError("cannot construct a wrong plan without a distractor card")

    omitted = min(
        deal.targets,
        key=lambda card: _control_rank(seed, "omit", card.name),
    )
    replacement = min(
        distractors,
        key=lambda card: _control_rank(seed, "replace", card.name),
    )
    selected = tuple(card for card in deal.targets if card.name != omitted.name) + (replacement,)
    return tuple(
        sorted(
            selected,
            key=lambda card: _control_rank(seed, "order", card.name),
        )
    )


def _control_rank(seed: int, purpose: str, card_name: str) -> bytes:
    material = f"vlabench-texas-wrong-plan-v1:{seed}:{purpose}:{card_name}"
    return hashlib.sha256(material.encode("utf-8")).digest()


def _activate_card_plan(
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
        metadata={
            "planner": planner_name,
            "seed": seed,
            "hand_type": deal.hand_type,
        },
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


def _advance_completed_plan_steps(
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


def _grasped_poker_name(endpoint: Any) -> str | None:
    task, physics = _task_and_physics(endpoint)
    robot = getattr(task, "robot", None)
    if robot is None:
        raise RuntimeError("texas_holdem task did not expose its robot")
    for poker in tuple(getattr(task, "pokers", ())):
        is_grasped = getattr(poker, "is_grasped", None)
        if callable(is_grasped) and bool(is_grasped(physics, robot)):
            return str(poker.name)
    return None


def _contained_target_names(endpoint: Any, deal: PokerDeal) -> frozenset[str]:
    return _contained_card_names(endpoint, deal.target_names)


def _contained_card_names(
    endpoint: Any,
    card_names: Sequence[str],
) -> frozenset[str]:
    task, physics = _task_and_physics(endpoint)
    entities = getattr(task, "entities", None)
    if not isinstance(entities, Mapping):
        raise TypeError("texas_holdem task did not expose task.entities as a mapping")
    container = entities.get(_TARGET_CONTAINER_NAME)
    contain = getattr(container, "contain", None)
    if not callable(contain):
        raise TypeError("Texas Hold'em placemat did not expose contain()")

    completed: set[str] = set()
    for name in card_names:
        entity = entities.get(name)
        get_xpos = getattr(entity, "get_xpos", None)
        if not callable(get_xpos):
            raise TypeError(f"Texas Hold'em target {name!r} did not expose get_xpos()")
        if bool(contain(get_xpos(physics), physics)):
            completed.add(name)
    return frozenset(completed)


def _task_and_physics(endpoint: Any) -> tuple[Any, Any]:
    environment = endpoint.raw_environment
    inner = getattr(environment, "_env", None)
    task = getattr(inner, "task", None)
    physics = getattr(inner, "physics", None)
    if task is None or physics is None:
        raise RuntimeError("VLABench endpoint did not expose a live task and physics")
    return task, physics


def _place_target_robot_frame(
    endpoint: Any,
    *,
    placement_index: int,
    config: EndpointPlaceControllerConfig,
) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("the VLABench place controller requires NumPy") from error

    environment = endpoint.raw_environment
    task, physics = _task_and_physics(endpoint)
    entities = getattr(task, "entities", None)
    if not isinstance(entities, Mapping):
        raise TypeError("texas_holdem task did not expose task.entities as a mapping")
    container = entities.get(_TARGET_CONTAINER_NAME)
    get_place_point = getattr(container, "get_place_point", None)
    if not callable(get_place_point):
        raise TypeError("Texas Hold'em placemat did not expose get_place_point()")
    place_points = list(get_place_point(physics))
    if not place_points:
        raise RuntimeError("Texas Hold'em placemat did not return a place point")
    target_world = np.asarray(place_points[-1], dtype=np.float64).reshape(3).copy()
    slot = placement_index % config.slot_count
    target_world[1] += (slot - (config.slot_count - 1) / 2.0) * config.slot_spacing_m

    robot = getattr(task, "robot", None)
    ee_offset = getattr(robot, "ee_offset", None)
    if not callable(ee_offset):
        raise TypeError("VLABench robot did not expose ee_offset()")
    target_world += np.asarray(ee_offset(physics), dtype=np.float64).reshape(3)

    base = getattr(environment, "_robot_base_xyz", None)
    if base is None:
        get_robot_frame_position = getattr(
            getattr(environment, "_env", None), "get_robot_frame_position", None
        )
        if not callable(get_robot_frame_position):
            raise RuntimeError("VLABench environment did not expose its robot base position")
        base = get_robot_frame_position()
    target_robot = target_world - np.asarray(base, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(target_robot)):
        raise RuntimeError("place target contains non-finite coordinates")
    if np.max(np.abs(target_robot)) > config.workspace_abs_limit_m:
        raise RuntimeError(
            f"place target {target_robot.tolist()} exceeds configured workspace limit"
        )
    return target_robot


def _agent_state(observation: RobotObservation) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("the VLABench place controller requires NumPy") from error
    if not isinstance(observation.values, Mapping) or "agent_pos" not in observation.values:
        raise RuntimeError("VLABench observation did not contain agent_pos")
    state = np.asarray(observation.values["agent_pos"], dtype=np.float64).reshape(-1)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise RuntimeError("VLABench agent_pos must be a finite 7-D vector")
    return state


def _build_place_waypoints(
    state: Any,
    target_position: Any,
    *,
    config: EndpointPlaceControllerConfig,
) -> list[tuple[str, Any]]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("the VLABench place controller requires NumPy") from error

    state = np.asarray(state, dtype=np.float64).reshape(7)
    target = np.asarray(target_position, dtype=np.float64).reshape(3)
    current = state[:3].copy()
    euler = state[3:6].copy()
    safe_z = max(
        current[2] + config.lift_height_m,
        target[2] + config.clearance_m,
    )
    lift = np.array([current[0], current[1], safe_z], dtype=np.float64)
    above = np.array([target[0], target[1], safe_z], dtype=np.float64)
    retract = np.array(
        [target[0], target[1], max(safe_z, target[2] + config.retract_height_m)],
        dtype=np.float64,
    )

    result: list[tuple[str, Any]] = []

    def add_segment(phase: str, destination: Any, gripper: float) -> None:
        nonlocal current
        destination = np.asarray(destination, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(destination - current))
        count = max(1, math.ceil(distance / config.max_translation_step_m))
        start = current.copy()
        for fraction in np.linspace(1.0 / count, 1.0, count):
            position = start + (destination - start) * float(fraction)
            result.append(
                (
                    phase,
                    np.concatenate([position, euler, np.array([gripper])]).astype(np.float32),
                )
            )
        current = destination

    add_segment("lift", lift, 0.0)
    add_segment("translate", above, 0.0)
    add_segment("descend", target, 0.0)
    open_action = np.concatenate([target, euler, np.array([1.0])]).astype(np.float32)
    result.extend(("release", open_action.copy()) for _ in range(config.open_steps))
    add_segment("retract", retract, 1.0)
    return result


def _deal_document(deal: PokerDeal) -> dict[str, Any]:
    return {
        "cards": [asdict(card) for card in deal.cards],
        "target_names": list(deal.target_names),
        "target_prompts": [card.primitive_prompt for card in deal.targets],
        "hand_type": deal.hand_type,
    }


def _seed_policy(seed: int) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("the VLABench policy environment requires PyTorch") from error
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_runner(*args: Any, **kwargs: Any) -> Any:
    from embodied_runtime.integrations.lerobot import VLABenchSmolVLARunner

    return VLABenchSmolVLARunner(*args, **kwargs)


def _write_outputs(
    *,
    config: TexasHoldemExperimentConfig,
    report: EvaluationReport,
    traces: Sequence[Mapping[str, Any]],
    runtime_facts: Any,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    gate_config = OracleGateConfig()
    plan_quality_control = paired_binary_comparison(
        report.records,
        baseline=ExperimentCondition.WRONG_PLAN,
        comparator=ExperimentCondition.ORACLE_PLAN,
    )
    (config.output_dir / "report.json").write_text(
        report.to_json(oracle_gate_config=gate_config),
        encoding="utf-8",
    )
    (config.output_dir / "trials.csv").write_text(report.trials_csv(), encoding="utf-8")
    (config.output_dir / "summaries.csv").write_text(
        report.summaries_csv(),
        encoding="utf-8",
    )
    (config.output_dir / "comparisons.csv").write_text(
        report.comparisons_csv(),
        encoding="utf-8",
    )
    (config.output_dir / "plan_control_comparisons.csv").write_text(
        report.comparisons_csv(baseline=ExperimentCondition.WRONG_PLAN),
        encoding="utf-8",
    )
    (config.output_dir / "plan_quality_control.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hypothesis": (
                    "With the same primitive skill executor, the exact target plan "
                    "should outperform a plan with one target replaced by a distractor."
                ),
                "comparison": plan_quality_control.to_dict(),
                "interpretation": (
                    "A wrong-plan success or no oracle advantage is a diagnostic "
                    "failure; it is not evidence that planning is unnecessary."
                ),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "traces.jsonl").write_text(
        "".join(
            json.dumps(document, allow_nan=False, sort_keys=True) + "\n" for document in traces
        ),
        encoding="utf-8",
    )
    facts_document = (
        asdict(runtime_facts)
        if runtime_facts is not None and hasattr(runtime_facts, "__dataclass_fields__")
        else None
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": config.checkpoint,
                "task": config.task,
                "device": config.device,
                "backbone_path": config.backbone_path,
                "seeds": list(config.seeds),
                "max_episode_steps": config.max_episode_steps,
                "control_period_s": config.control_period_s,
                "warmup_policy": config.warmup_policy,
                "local_files_only": config.local_files_only,
                "composite_prompt": TEXAS_HOLDEM_COMPOSITE_PROMPT,
                "place_controller": asdict(config.place_controller),
                "runtime_facts": facts_document,
                "claim_scope": (
                    "matched task-level card-selection effect for one fixed "
                    "checkpoint using exact VLABench training labels and an "
                    "equal-length wrong-plan negative control"
                ),
                "oracle_caveat": (
                    "oracle_plan reads simulator target identities and is an "
                    "upper bound, not a deployable cloud planner"
                ),
                "negative_control": (
                    "wrong_plan also reads simulator target identities solely to "
                    "replace exactly one target with a distractor; it uses the same "
                    "PlanManager, primitive prompt path, runner, place controller, "
                    "seed, initial scene, and action budget as oracle_plan"
                ),
                "controller_validation": (
                    "validate clearance and release height on each downloaded "
                    "placemat asset before the formal multi-seed run"
                ),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
        description=("Run matched VLABench Texas Hold'em Edge/Oracle/Wrong-plan controls.")
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(1000,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone-path",
        help="optional local SmolVLM2 snapshot used for deterministic offline loading",
    )
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--control-period-s", type=float, default=0.1)
    parser.add_argument("--place-lift-height-m", type=float, default=0.15)
    parser.add_argument("--place-clearance-m", type=float, default=0.12)
    parser.add_argument("--place-retract-height-m", type=float, default=0.10)
    parser.add_argument("--place-max-step-m", type=float, default=0.025)
    parser.add_argument("--place-open-steps", type=int, default=5)
    parser.add_argument("--place-slot-count", type=int, default=5)
    parser.add_argument("--place-slot-spacing-m", type=float, default=0.05)
    parser.add_argument("--place-workspace-limit-m", type=float, default=2.0)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="skip the unmeasured policy warm-up before the first paired trial",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow missing model files to download",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TexasHoldemExperimentConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        seeds=args.seeds,
        device=args.device,
        backbone_path=args.backbone_path,
        max_episode_steps=args.max_episode_steps,
        control_period_s=args.control_period_s,
        warmup_policy=not args.skip_warmup,
        local_files_only=not args.allow_download,
        place_controller=EndpointPlaceControllerConfig(
            lift_height_m=args.place_lift_height_m,
            clearance_m=args.place_clearance_m,
            retract_height_m=args.place_retract_height_m,
            max_translation_step_m=args.place_max_step_m,
            open_steps=args.place_open_steps,
            slot_count=args.place_slot_count,
            slot_spacing_m=args.place_slot_spacing_m,
            workspace_abs_limit_m=args.place_workspace_limit_m,
        ),
    )
    report = run_texas_holdem_experiment(config)
    print(report.to_json(oracle_gate_config=OracleGateConfig()))
    return 0


__all__ = [
    "TEXAS_HOLDEM_COMPOSITE_PROMPT",
    "TEXAS_HOLDEM_TASK",
    "EndpointPlaceController",
    "EndpointPlaceControllerConfig",
    "PokerCard",
    "PokerDeal",
    "TexasHoldemExperimentConfig",
    "build_parser",
    "main",
    "run_texas_holdem_experiment",
]
