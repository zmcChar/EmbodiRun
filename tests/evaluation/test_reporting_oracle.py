from __future__ import annotations

from collections.abc import Callable

import pytest

from embodied_runtime.evaluation.reporting import (
    ExperimentCondition,
    OracleGateConfig,
    TrialRecord,
    evaluate_oracle_gate,
)


def test_oracle_gate_requires_gain_significance_sample_size_and_complete_pairs(
    trial_factory: Callable[..., TrialRecord],
) -> None:
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


def test_gate_configuration_is_strict() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        OracleGateConfig(minimum_gain_percentage_points=101)
    with pytest.raises(ValueError, match="greater than zero"):
        OracleGateConfig(minimum_paired_trials=0)
    with pytest.raises(ValueError, match="at most 1"):
        OracleGateConfig(maximum_p_value=2)
    with pytest.raises(TypeError, match="bool"):
        OracleGateConfig(require_complete_pairs=1)
