"""Paired VLABench pilot for testing task-level planning over a fixed edge VLA.

The first gate intentionally compares only two conditions:

``edge_only``
    SmolVLA receives the original composite instruction for the whole episode.

``oracle_plan``
    The same SmolVLA receives an oracle decomposition, with its cached action
    chunk cleared at the subgoal boundary.

Only if the oracle condition improves paired success should a learned cloud
planner be introduced.  This isolates the value of decomposition from planner
quality and distributed-system effects.
"""

from __future__ import annotations

import argparse
import json
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
)
from embodied_runtime.simulators import VLABenchSimulatorEndpoint

GET_COFFEE_TASK = "get_coffee"
GET_COFFEE_COMPOSITE_PROMPT = "composite: Get me a cup of coffee."
GET_COFFEE_ORACLE_STEPS = (
    PlanStep(
        step_id="place_mug",
        skill="place_mug",
        instruction=("primitive: Pick up the mug and place it under the coffee dispenser."),
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
        for name in (
            "max_episode_steps",
            "first_subgoal_budget",
            "subgoal_stability_steps",
        ):
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


@dataclass(frozen=True, slots=True)
class _EpisodeRun:
    record: TrialRecord
    events: tuple[dict[str, Any], ...]


def run_vlabench_pilot(
    config: VLABenchPilotConfig,
    *,
    runner_factory: Callable[..., Any] | None = None,
    endpoint_factory: Callable[..., Any] | None = None,
    policy_seeder: Callable[[int], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    """Run paired conditions, validate resets, and persist an auditable report."""

    if not isinstance(config, VLABenchPilotConfig):
        raise TypeError("config must be a VLABenchPilotConfig")
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

    records: list[TrialRecord] = []
    trace_documents: list[dict[str, Any]] = []
    for pair_index, seed in enumerate(config.seeds):
        endpoint = endpoint_factory(
            config.task,
            max_episode_steps=config.max_episode_steps,
        )
        # Alternate order so one condition does not own every cold/warm episode.
        conditions = (
            (
                ExperimentCondition.EDGE_ONLY,
                ExperimentCondition.ORACLE_PLAN,
            )
            if pair_index % 2 == 0
            else (
                ExperimentCondition.ORACLE_PLAN,
                ExperimentCondition.EDGE_ONLY,
            )
        )
        pair_runs: list[_EpisodeRun] = []
        try:
            if pair_index == 0 and config.warmup_policy:
                warmup_observation = endpoint.reset(seed=seed)
                policy_seeder(seed)
                runner.reset_action_queue()
                runner.select_action(
                    warmup_observation.values,
                    GET_COFFEE_COMPOSITE_PROMPT,
                )
                runner.reset_action_queue()
            for condition in conditions:
                pair_runs.append(
                    _run_episode(
                        config=config,
                        endpoint=endpoint,
                        runner=runner,
                        condition=condition,
                        seed=seed,
                        policy_seeder=policy_seeder,
                        clock=clock,
                    )
                )
        finally:
            endpoint.close()

        fingerprints = tuple(run.record.initial_fingerprint for run in pair_runs)
        if len(fingerprints) != 2 or any(value is None for value in fingerprints):
            raise RuntimeError(f"paired seed {seed} is missing an initial observation fingerprint")
        if fingerprints[0] != fingerprints[1]:
            raise RuntimeError(f"paired seed {seed} did not reproduce the same initial observation")
        for run in pair_runs:
            records.append(run.record)
            trace_documents.append(
                {
                    "pair_id": run.record.pair_id,
                    "task_id": run.record.task_id,
                    "seed": run.record.seed,
                    "condition": run.record.condition.value,
                    "initial_fingerprint": run.record.initial_fingerprint,
                    "events": list(run.events),
                }
            )
        # Persist every completed pair. A later simulator, IK, or GPU failure
        # must not discard hours of already completed paired trials.
        _write_outputs(
            config=config,
            report=EvaluationReport(records),
            traces=trace_documents,
            runtime_facts=getattr(runner, "facts", None),
        )

    report = EvaluationReport(records)
    return report


def _run_episode(
    *,
    config: VLABenchPilotConfig,
    endpoint: Any,
    runner: Any,
    condition: ExperimentCondition,
    seed: int,
    policy_seeder: Callable[[int], None],
    clock: Callable[[], float],
) -> _EpisodeRun:
    observation: RobotObservation = endpoint.reset(seed=seed)
    fingerprint = endpoint.initial_fingerprint
    policy_seeder(seed)
    runner.reset_action_queue()

    plan_manager: PlanManager | None = None
    if condition is ExperimentCondition.ORACLE_PLAN:
        plan_manager = _activate_oracle_plan(
            observation=observation,
            fingerprint=fingerprint,
            seed=seed,
        )
        plan_manager.record_feedback(PlanStepStatus.ACTIVE)

    pair_id = f"{config.task}-seed-{seed:08d}"
    edge_latencies_ms: list[float] = []
    chunk_generation_latencies_ms: list[float] = []
    queue_hit_latencies_ms: list[float] = []
    deadline_misses = 0
    events: list[dict[str, Any]] = []
    subgoals_completed = 0
    contained_streak = 0
    success = False
    steps = 0

    for step_index in range(config.max_episode_steps):
        if condition is ExperimentCondition.EDGE_ONLY:
            prompt = GET_COFFEE_COMPOSITE_PROMPT
            plan_step_id = None
        else:
            assert plan_manager is not None
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
        model_latency_ms = selected.inference_latency_s * 1000.0
        if bool(getattr(selected, "generated_chunk", False)):
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
                },
            )
        )
        steps = step_index + 1
        observation = outcome.observation
        event = {
            "step": steps,
            "prompt": prompt,
            "plan_step_id": plan_step_id,
            "edge_latency_ms": latency_ms,
            "model_latency_ms": model_latency_ms,
            "generated_chunk": bool(getattr(selected, "generated_chunk", False)),
            "queue_remaining": getattr(selected, "queue_remaining", None),
            "success": bool(outcome.success),
            "physics_error": bool(outcome.info.get("physics_error", False)),
        }

        if outcome.success:
            success = True
            subgoals_completed = 2
            if plan_manager is not None and plan_manager.current_step is not None:
                plan_manager.advance(PlanStepStatus.SUCCEEDED)
            event["subgoal_completed"] = "press_start"
            events.append(event)
            break

        # Upstream VLABench drops its inner environment after a PhysicsError.
        # Never inspect task conditions once a terminal outcome has invalidated
        # that environment.
        if outcome.done:
            events.append(event)
            break

        if subgoals_completed == 0:
            contained_streak = contained_streak + 1 if _coffee_mug_is_placed(endpoint) else 0
        queue_remaining = getattr(selected, "queue_remaining", None)
        at_action_chunk_boundary = queue_remaining in (None, 0)
        if (
            subgoals_completed == 0
            and contained_streak >= config.subgoal_stability_steps
            and at_action_chunk_boundary
        ):
            subgoals_completed = 1
            event["subgoal_completed"] = "place_mug"
            event["contained_streak"] = contained_streak
            if plan_manager is not None:
                plan_manager.advance(PlanStepStatus.SUCCEEDED)
                if plan_manager.current_step is not None:
                    plan_manager.record_feedback(PlanStepStatus.ACTIVE)
                # The policy may still hold actions from the previous 50-step
                # chunk.  They must never cross a semantic plan boundary.
                runner.reset_action_queue()

        events.append(event)
        if (
            condition is ExperimentCondition.ORACLE_PLAN
            and subgoals_completed == 0
            and steps == config.first_subgoal_budget
        ):
            events.append(
                {
                    "step": steps,
                    "event": "first_subgoal_budget_reached",
                    "plan_step_id": "place_mug",
                }
            )

    record = TrialRecord(
        pair_id=pair_id,
        task_id=config.task,
        seed=seed,
        condition=condition,
        success=success,
        steps=steps,
        edge_latencies_ms=tuple(edge_latencies_ms),
        planner_latencies_ms=(),
        chunk_generation_latencies_ms=tuple(chunk_generation_latencies_ms),
        queue_hit_latencies_ms=tuple(queue_hit_latencies_ms),
        deadline_misses=deadline_misses,
        subgoals_completed=subgoals_completed,
        subgoals_total=2,
        initial_fingerprint=fingerprint,
    )
    return _EpisodeRun(record=record, events=tuple(events))


def _activate_oracle_plan(
    *,
    observation: RobotObservation,
    fingerprint: str,
    seed: int,
) -> PlanManager:
    session_id = f"get-coffee-{seed}"
    goal = TaskGoal(
        task_id=GET_COFFEE_TASK,
        session_id=session_id,
        instruction=GET_COFFEE_COMPOSITE_PROMPT,
        allowed_skills=("place_mug", "press_start"),
        metadata={"planner": "oracle", "seed": seed},
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
        steps=GET_COFFEE_ORACLE_STEPS,
        created_at_s=created_at_s,
        expires_at_s=created_at_s + 3600.0,
        based_on_observation_id=fingerprint,
        metadata={"planner": "oracle"},
    )
    manager = PlanManager(session_id)
    manager.offer_plan(envelope, request=request)
    activated = manager.activate_pending(safe_boundary=True)
    if activated is None:
        raise RuntimeError("oracle plan could not be activated at episode reset")
    return manager


def _coffee_mug_is_placed(endpoint: Any) -> bool:
    environment = endpoint.raw_environment
    inner = getattr(environment, "_env", None)
    task = getattr(inner, "task", None)
    physics = getattr(inner, "physics", None)
    conditions = getattr(getattr(task, "conditions", None), "conditions", None)
    if not conditions:
        raise RuntimeError("get_coffee does not expose its contain condition")
    is_met = getattr(conditions[0], "is_met", None)
    if not callable(is_met):
        raise TypeError("get_coffee contain condition does not expose is_met()")
    return bool(is_met(physics))


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
    config: VLABenchPilotConfig,
    report: EvaluationReport,
    traces: Sequence[Mapping[str, Any]],
    runtime_facts: Any,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    gate_config = OracleGateConfig()
    (config.output_dir / "report.json").write_text(
        report.to_json(oracle_gate_config=gate_config),
        encoding="utf-8",
    )
    (config.output_dir / "trials.csv").write_text(
        report.trials_csv(),
        encoding="utf-8",
    )
    (config.output_dir / "summaries.csv").write_text(
        report.summaries_csv(),
        encoding="utf-8",
    )
    (config.output_dir / "comparisons.csv").write_text(
        report.comparisons_csv(),
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
                "first_subgoal_budget": config.first_subgoal_budget,
                "subgoal_stability_steps": config.subgoal_stability_steps,
                "control_period_s": config.control_period_s,
                "warmup_policy": config.warmup_policy,
                "local_files_only": config.local_files_only,
                "runtime_facts": facts_document,
                "claim_scope": (
                    "paired prompt-decomposition effect for one fixed checkpoint; "
                    "not unseen-composition generalization"
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
        description="Run the paired VLABench Edge-only/Oracle-plan gate."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(1000,))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone-path",
        help="optional local SmolVLM2 snapshot used for deterministic offline loading",
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--first-subgoal-budget", type=int, default=350)
    parser.add_argument("--subgoal-stability-steps", type=int, default=3)
    parser.add_argument("--control-period-s", type=float, default=0.1)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="skip the unmeasured policy warm-up before the first paired trial",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model files missing from the local checkpoint/cache to download",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VLABenchPilotConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        seeds=args.seeds,
        device=args.device,
        backbone_path=args.backbone_path,
        max_episode_steps=args.max_episode_steps,
        first_subgoal_budget=args.first_subgoal_budget,
        subgoal_stability_steps=args.subgoal_stability_steps,
        control_period_s=args.control_period_s,
        warmup_policy=not args.skip_warmup,
        local_files_only=not args.allow_download,
    )
    report = run_vlabench_pilot(config)
    print(report.to_json(oracle_gate_config=OracleGateConfig()))
    return 0


__all__ = [
    "GET_COFFEE_COMPOSITE_PROMPT",
    "GET_COFFEE_ORACLE_STEPS",
    "GET_COFFEE_TASK",
    "VLABenchPilotConfig",
    "build_parser",
    "main",
    "run_vlabench_pilot",
]
