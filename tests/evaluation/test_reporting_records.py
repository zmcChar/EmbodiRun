from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from embodied_runtime.evaluation.reporting import (
    EvaluationReport,
    ExperimentCondition,
    TrialRecord,
)


def test_trial_record_strictly_validates_identity_metrics_and_counts(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    record = TrialRecord(
        pair_id=" pair-1 ",
        task_id=" task ",
        seed=1,
        condition="edge_only",
        success=False,
        steps=0,
        edge_latencies_ms=[1, 2.5],
        deadline_misses=1,
    )

    assert record.pair_id == "pair-1"
    assert record.task_id == "task"
    assert record.condition is ExperimentCondition.EDGE_ONLY
    assert record.edge_latencies_ms == (1.0, 2.5)

    with pytest.raises(ValueError, match="pair_id"):
        trial_factory(1, ExperimentCondition.EDGE_ONLY, False).__class__(
            "", "task", 1, ExperimentCondition.EDGE_ONLY, False, 1
        )
    with pytest.raises(TypeError, match="seed"):
        TrialRecord("pair", "task", True, ExperimentCondition.EDGE_ONLY, False, 1)
    with pytest.raises(ValueError, match="unsupported"):
        TrialRecord("pair", "task", 1, "unknown", False, 1)
    with pytest.raises(TypeError, match="success"):
        TrialRecord("pair", "task", 1, ExperimentCondition.EDGE_ONLY, 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        TrialRecord("pair", "task", 1, ExperimentCondition.EDGE_ONLY, False, -1)
    with pytest.raises(ValueError, match="finite"):
        TrialRecord(
            "pair",
            "task",
            1,
            ExperimentCondition.EDGE_ONLY,
            False,
            1,
            (math.inf,),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        TrialRecord(
            "pair",
            "task",
            1,
            ExperimentCondition.EDGE_ONLY,
            False,
            1,
            (1.0,),
            deadline_misses=2,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        TrialRecord(
            pair_id="bad",
            task_id="task",
            seed=3,
            condition=ExperimentCondition.ORACLE_PLAN,
            success=False,
            steps=1,
            subgoals_completed=2,
            subgoals_total=1,
        )


def test_report_rejects_duplicate_or_inconsistent_pairs(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    record = trial_factory(1, ExperimentCondition.EDGE_ONLY, False)
    with pytest.raises(ValueError, match="at most one"):
        EvaluationReport([record, record])

    mismatched = TrialRecord(
        pair_id=record.pair_id,
        task_id="different-task",
        seed=record.seed,
        condition=ExperimentCondition.ORACLE_PLAN,
        success=True,
        steps=1,
    )
    with pytest.raises(ValueError, match="same task_id and seed"):
        EvaluationReport([record, mismatched])

    first = TrialRecord(
        pair_id="fingerprint-pair",
        task_id="task",
        seed=2,
        condition=ExperimentCondition.EDGE_ONLY,
        success=False,
        steps=1,
        initial_fingerprint="first",
    )
    second = TrialRecord(
        pair_id="fingerprint-pair",
        task_id="task",
        seed=2,
        condition=ExperimentCondition.ORACLE_PLAN,
        success=False,
        steps=1,
        initial_fingerprint="second",
    )
    with pytest.raises(ValueError, match="same observation fingerprint"):
        EvaluationReport([first, second])
