"""Matched Texas Hold'em experiment composition and seed pairing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from embodied_runtime.evaluation import EvaluationReport, ExperimentCondition, TrialRecord
from embodied_runtime.simulators import VLABenchSimulatorEndpoint

from ..runtime import make_smolvla_runner, seed_policy
from .place_controller import EndpointPlaceController
from .planning import deal_document
from .reporting import write_outputs
from .settings import (
    TEXAS_HOLDEM_COMPOSITE_PROMPT,
    TexasHoldemExperimentConfig,
)
from .trial import EpisodeRun, run_episode


def rotated_conditions(pair_index: int) -> tuple[ExperimentCondition, ...]:
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
    place_controller = EndpointPlaceController(config.place_controller)

    records: list[TrialRecord] = []
    trace_documents: list[dict[str, Any]] = []
    for pair_index, seed in enumerate(config.seeds):
        endpoint = endpoint_factory(config.task, max_episode_steps=config.max_episode_steps)
        conditions = rotated_conditions(pair_index)
        pair_runs: list[EpisodeRun] = []
        try:
            if pair_index == 0 and config.warmup_policy:
                warmup_observation = endpoint.reset(seed=seed)
                policy_seeder(seed)
                runner.reset_action_queue()
                runner.select_action(warmup_observation.values, TEXAS_HOLDEM_COMPOSITE_PROMPT)
                runner.reset_action_queue()
            for condition in conditions:
                pair_runs.append(
                    run_episode(
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
                    "deal": deal_document(run.deal),
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


__all__ = ["run_texas_holdem_experiment"]
