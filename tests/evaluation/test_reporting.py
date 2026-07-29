from __future__ import annotations

import csv
import io
import json
import math

import pytest

from embodied_runtime.evaluation import (
    EvaluationReport,
    ExperimentCondition,
    OracleGateConfig,
    TrialRecord,
    evaluate_oracle_gate,
    paired_binary_comparison,
    summarize_condition,
)


def _trial(
    pair: int,
    condition: ExperimentCondition,
    success: bool,
    *,
    edge: tuple[float, ...] = (1.0,),
    planner: tuple[float, ...] = (),
    chunks: tuple[float, ...] = (),
    queue_hits: tuple[float, ...] = (),
    deadline_misses: int = 0,
) -> TrialRecord:
    return TrialRecord(
        pair_id=f"pair-{pair:02d}",
        task_id="composite-pick",
        seed=pair,
        condition=condition,
        success=success,
        steps=10 + pair,
        edge_latencies_ms=edge,
        planner_latencies_ms=planner,
        chunk_generation_latencies_ms=chunks,
        queue_hit_latencies_ms=queue_hits,
        deadline_misses=deadline_misses,
    )


def test_trial_record_strictly_validates_identity_metrics_and_counts() -> None:
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
        _trial(1, ExperimentCondition.EDGE_ONLY, False).__class__(
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


def test_report_rejects_duplicate_or_inconsistent_pairs() -> None:
    record = _trial(1, ExperimentCondition.EDGE_ONLY, False)
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


def test_condition_summary_reports_wilson_latency_steps_and_deadlines() -> None:
    records = [
        _trial(
            1,
            ExperimentCondition.CLOSED_LOOP,
            True,
            edge=(1.0, 2.0),
            planner=(100.0,),
            chunks=(20.0,),
            queue_hits=(0.2, 0.3),
            deadline_misses=1,
        ),
        _trial(
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


def test_subgoal_progress_is_aggregated_and_validated() -> None:
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


def test_empty_latency_streams_are_reported_as_missing_not_zero() -> None:
    summary = summarize_condition([_trial(1, ExperimentCondition.EDGE_ONLY, False, edge=())])

    assert summary.edge_calls == 0
    assert summary.edge_latency_ms_mean is None
    assert summary.edge_latency_ms_p99 is None
    assert summary.planner_calls == 0
    assert summary.planner_latency_ms_p50 is None
    assert summary.deadline_miss_rate is None


def test_exact_paired_comparison_counts_discordance_and_effect() -> None:
    # Eight Oracle-only successes, one Edge-only success, and one shared success.
    edge_results = [False] * 8 + [True, True]
    oracle_results = [True] * 8 + [False, True]
    records = [
        trial
        for pair, (edge_success, oracle_success) in enumerate(
            zip(edge_results, oracle_results, strict=True)
        )
        for trial in (
            _trial(pair, ExperimentCondition.EDGE_ONLY, edge_success),
            _trial(pair, ExperimentCondition.ORACLE_PLAN, oracle_success),
        )
    ]

    comparison = paired_binary_comparison(
        records,
        comparator=ExperimentCondition.ORACLE_PLAN,
    )

    assert comparison.matched_pairs == 10
    assert comparison.both_succeeded == 1
    assert comparison.both_failed == 0
    assert comparison.baseline_only_succeeded == 1
    assert comparison.comparator_only_succeeded == 8
    assert comparison.discordant_pairs == 9
    assert comparison.effect_percentage_points == 70.0
    # 2 * (C(9, 0) + C(9, 1)) / 2**9
    assert comparison.exact_two_sided_p_value == 0.0390625


def test_paired_comparison_reports_unmatched_trials_without_using_them() -> None:
    records = [
        _trial(1, ExperimentCondition.EDGE_ONLY, False),
        _trial(2, ExperimentCondition.EDGE_ONLY, True),
        _trial(1, ExperimentCondition.OPEN_LOOP, True),
        _trial(3, ExperimentCondition.OPEN_LOOP, True),
    ]

    comparison = paired_binary_comparison(
        records,
        comparator=ExperimentCondition.OPEN_LOOP,
    )

    assert comparison.matched_pairs == 1
    assert comparison.baseline_unmatched_trials == 1
    assert comparison.comparator_unmatched_trials == 1
    assert comparison.effect_percentage_points == 100.0
    assert comparison.exact_two_sided_p_value == 1.0


def test_report_can_render_wrong_plan_as_the_explicit_control_baseline() -> None:
    records = [
        _trial(1, ExperimentCondition.WRONG_PLAN, False),
        _trial(1, ExperimentCondition.ORACLE_PLAN, True),
        _trial(2, ExperimentCondition.WRONG_PLAN, False),
        _trial(2, ExperimentCondition.ORACLE_PLAN, False),
    ]
    report = EvaluationReport(records)

    comparisons = report.comparisons(baseline=ExperimentCondition.WRONG_PLAN)
    assert len(comparisons) == 1
    assert comparisons[0].baseline is ExperimentCondition.WRONG_PLAN
    assert comparisons[0].comparator is ExperimentCondition.ORACLE_PLAN
    assert comparisons[0].effect_percentage_points == 50.0

    rows = list(
        csv.DictReader(io.StringIO(report.comparisons_csv(baseline=ExperimentCondition.WRONG_PLAN)))
    )
    assert rows[0]["baseline"] == "wrong_plan"
    assert rows[0]["comparator"] == "oracle_plan"


def test_oracle_gate_requires_gain_significance_sample_size_and_complete_pairs() -> None:
    edge_results = [False] * 8 + [True, True]
    oracle_results = [True] * 8 + [False, True]
    records = [
        trial
        for pair, (edge_success, oracle_success) in enumerate(
            zip(edge_results, oracle_results, strict=True)
        )
        for trial in (
            _trial(pair, ExperimentCondition.EDGE_ONLY, edge_success),
            _trial(pair, ExperimentCondition.ORACLE_PLAN, oracle_success),
        )
    ]
    config = OracleGateConfig(
        minimum_gain_percentage_points=50,
        minimum_paired_trials=10,
        minimum_discordant_pairs=5,
        maximum_p_value=0.04,
    )

    passed = evaluate_oracle_gate(records, config=config)
    assert passed.passed is True
    assert passed.failed_checks == ()

    incomplete = records[:-1]
    failed = evaluate_oracle_gate(incomplete, config=config)
    assert failed.passed is False
    assert "insufficient_paired_trials" in failed.failed_checks
    assert "incomplete_pairing" in failed.failed_checks


def test_report_json_and_csv_are_deterministic_and_machine_readable() -> None:
    records = [
        _trial(
            2,
            ExperimentCondition.ORACLE_PLAN,
            True,
            edge=(2.0,),
            planner=(8.0,),
        ),
        _trial(1, ExperimentCondition.EDGE_ONLY, False, edge=(1.0,)),
        _trial(2, ExperimentCondition.EDGE_ONLY, False, edge=(3.0,)),
        _trial(
            1,
            ExperimentCondition.ORACLE_PLAN,
            True,
            edge=(4.0,),
            planner=(9.0,),
        ),
    ]
    report = EvaluationReport(records)
    reversed_report = EvaluationReport(reversed(records))
    gate = OracleGateConfig(
        minimum_gain_percentage_points=0,
        minimum_paired_trials=1,
        maximum_p_value=1,
    )

    assert report.to_json(oracle_gate_config=gate) == reversed_report.to_json(
        oracle_gate_config=gate
    )
    document = json.loads(report.to_json(oracle_gate_config=gate))
    assert document["schema_version"] == 1
    assert [trial["condition"] for trial in document["trials"]] == [
        "edge_only",
        "edge_only",
        "oracle_plan",
        "oracle_plan",
    ]
    assert document["oracle_gate"]["passed"] is True

    for rendered in (
        report.trials_csv(),
        report.summaries_csv(),
        report.comparisons_csv(),
    ):
        assert rendered.endswith("\n")
        assert list(csv.reader(io.StringIO(rendered)))
        assert "\r\n" not in rendered

    trial_rows = list(csv.DictReader(io.StringIO(report.trials_csv())))
    assert trial_rows[0]["pair_id"] == "pair-01"
    assert json.loads(trial_rows[0]["edge_latencies_ms"]) == [1.0]
    assert trial_rows[0]["success"] == "0"


def test_gate_configuration_is_strict() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        OracleGateConfig(minimum_gain_percentage_points=101)
    with pytest.raises(ValueError, match="greater than zero"):
        OracleGateConfig(minimum_paired_trials=0)
    with pytest.raises(ValueError, match="at most 1"):
        OracleGateConfig(maximum_p_value=2)
    with pytest.raises(TypeError, match="bool"):
        OracleGateConfig(require_complete_pairs=1)
