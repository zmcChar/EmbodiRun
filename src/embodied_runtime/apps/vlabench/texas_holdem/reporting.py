"""Auditable JSON/JSONL/CSV output for Texas Hold'em trials."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from embodied_runtime.evaluation import (
    EvaluationReport,
    ExperimentCondition,
    OracleGateConfig,
    paired_binary_comparison,
)

from .settings import TEXAS_HOLDEM_COMPOSITE_PROMPT, TexasHoldemExperimentConfig


def write_outputs(
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
