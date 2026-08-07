"""Planner skill-gate comparison and report output."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .settings import (
    PlannerSkillCondition,
    PlannerSkillGateConfig,
    PlannerSkillTrial,
)


def write_gate_report(
    config: PlannerSkillGateConfig,
    trials: Sequence[PlannerSkillTrial],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = [
        comparison
        for comparison in (
            matched_comparison(
                trials,
                baseline=PlannerSkillCondition.WRONG_PLAN,
                comparator=PlannerSkillCondition.ORACLE_PLAN,
            ),
            matched_comparison(
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


def matched_comparison(
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
