from __future__ import annotations

from collections.abc import Callable

import pytest

from embodied_runtime.evaluation.reporting import (
    ExperimentCondition,
    TrialRecord,
    summarize_condition,
)


def test_condition_summary_reports_wilson_latency_steps_and_deadlines(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    records = [
        trial_factory(
            1,
            ExperimentCondition.CLOSED_LOOP,
            True,
            edge=(1.0, 2.0),
            planner=(100.0,),
            chunks=(20.0,),
            queue_hits=(0.2, 0.3),
            deadline_misses=1,
        ),
        trial_factory(
            2,
            ExperimentCondition.CLOSED_LOOP,
            False,
            edge=(3.0, 4.0),
            planner=(200.0, 300.0),
            chunks=(30.0,),
            queue_hits=(0.4,),
        ),
    ]

    summary = summarize_condition(records)

    assert summary.trials == 2
    assert summary.successes == 1
    assert summary.success_rate == 0.5
    assert summary.success_rate_ci95_low == pytest.approx(0.0945312057)
    assert summary.success_rate_ci95_high == pytest.approx(0.9054687943)
    assert summary.steps_total == 23
    assert summary.steps_mean == 11.5
    assert summary.steps_p50 == 11.5
    assert summary.steps_p95 == pytest.approx(11.95)
    assert summary.edge_calls == 4
    assert summary.edge_latency_ms_mean == 2.5
    assert summary.edge_latency_ms_p50 == 2.5
    assert summary.edge_latency_ms_p95 == pytest.approx(3.85)
    assert summary.edge_latency_ms_p99 == pytest.approx(3.97)
    assert summary.planner_calls == 3
    assert summary.planner_latency_ms_mean == 200.0
    assert summary.planner_latency_ms_p95 == pytest.approx(290.0)
    assert summary.chunk_generations == 2
    assert summary.chunk_generation_latency_ms_mean == 25.0
    assert summary.queue_hits == 3
    assert summary.queue_hit_latency_ms_mean == pytest.approx(0.3)
    assert summary.deadline_misses == 1
    assert summary.deadline_miss_rate == 0.25
    assert summary.subgoal_completion_rate is None


def test_subgoal_progress_is_aggregated() -> None:
    records = [
        TrialRecord(
            pair_id="one",
            task_id="task",
            seed=1,
            condition=ExperimentCondition.ORACLE_PLAN,
            success=False,
            steps=2,
            subgoals_completed=1,
            subgoals_total=2,
        ),
        TrialRecord(
            pair_id="two",
            task_id="task",
            seed=2,
            condition=ExperimentCondition.ORACLE_PLAN,
            success=True,
            steps=2,
            subgoals_completed=2,
            subgoals_total=2,
        ),
    ]

    summary = summarize_condition(records)

    assert summary.subgoals_completed == 3
    assert summary.subgoals_total == 4
    assert summary.subgoal_completion_rate == 0.75


def test_empty_latency_streams_are_reported_as_missing_not_zero(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    summary = summarize_condition([trial_factory(1, ExperimentCondition.EDGE_ONLY, False, edge=())])

    assert summary.edge_calls == 0
    assert summary.edge_latency_ms_mean is None
    assert summary.edge_latency_ms_p99 is None
    assert summary.planner_calls == 0
    assert summary.planner_latency_ms_p50 is None
    assert summary.deadline_miss_rate is None
