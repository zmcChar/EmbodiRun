"""Paired get-coffee pilot composition and seed pairing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from embodied_runtime.evaluation import EvaluationReport, ExperimentCondition, TrialRecord
from embodied_runtime.simulators import VLABenchSimulatorEndpoint

from ..runtime import make_smolvla_runner, seed_policy
from .reporting import write_outputs
from .settings import GET_COFFEE_COMPOSITE_PROMPT, VLABenchPilotConfig
from .trial import EpisodeRun, run_episode


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
    runner_factory = runner_factory or make_smolvla_runner
    endpoint_factory = endpoint_factory or VLABenchSimulatorEndpoint
    policy_seeder = policy_seeder or seed_policy

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
        endpoint = endpoint_factory(config.task, max_episode_steps=config.max_episode_steps)
        conditions = (
            (ExperimentCondition.EDGE_ONLY, ExperimentCondition.ORACLE_PLAN)
            if pair_index % 2 == 0
            else (ExperimentCondition.ORACLE_PLAN, ExperimentCondition.EDGE_ONLY)
        )
        pair_runs: list[EpisodeRun] = []
        try:
            if pair_index == 0 and config.warmup_policy:
                warmup_observation = endpoint.reset(seed=seed)
                policy_seeder(seed)
                runner.reset_action_queue()
                runner.select_action(warmup_observation.values, GET_COFFEE_COMPOSITE_PROMPT)
                runner.reset_action_queue()
            for condition in conditions:
                pair_runs.append(
                    run_episode(
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
        write_outputs(
            config=config,
            report=EvaluationReport(records),
            traces=trace_documents,
            runtime_facts=getattr(runner, "facts", None),
        )

    return EvaluationReport(records)


__all__ = ["run_vlabench_pilot"]
