"""Auditable JSON/JSONL/CSV output for the get-coffee pilot."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from embodied_runtime.evaluation import EvaluationReport, OracleGateConfig

from .settings import VLABenchPilotConfig


def write_outputs(
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
    (config.output_dir / "trials.csv").write_text(report.trials_csv(), encoding="utf-8")
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
