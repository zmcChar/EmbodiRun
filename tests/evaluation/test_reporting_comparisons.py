from __future__ import annotations

import csv
import io
from collections.abc import Callable

from embodied_runtime.evaluation.reporting import (
    EvaluationReport,
    ExperimentCondition,
    TrialRecord,
    paired_binary_comparison,
)


def test_exact_paired_comparison_counts_discordance_and_effect(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    # Eight Oracle-only successes, one Edge-only success, and one shared success.
    edge_results = [False] * 8 + [True, True]
    oracle_results = [True] * 8 + [False, True]
    records = [
        trial
        for pair, (edge_success, oracle_success) in enumerate(
            zip(edge_results, oracle_results, strict=True)
        )
        for trial in (
            trial_factory(pair, ExperimentCondition.EDGE_ONLY, edge_success),
            trial_factory(pair, ExperimentCondition.ORACLE_PLAN, oracle_success),
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


def test_paired_comparison_reports_unmatched_trials_without_using_them(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    records = [
        trial_factory(1, ExperimentCondition.EDGE_ONLY, False),
        trial_factory(2, ExperimentCondition.EDGE_ONLY, True),
        trial_factory(1, ExperimentCondition.OPEN_LOOP, True),
        trial_factory(3, ExperimentCondition.OPEN_LOOP, True),
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


def test_report_can_render_wrong_plan_as_the_explicit_control_baseline(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    records = [
        trial_factory(1, ExperimentCondition.WRONG_PLAN, False),
        trial_factory(1, ExperimentCondition.ORACLE_PLAN, True),
        trial_factory(2, ExperimentCondition.WRONG_PLAN, False),
        trial_factory(2, ExperimentCondition.ORACLE_PLAN, False),
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
