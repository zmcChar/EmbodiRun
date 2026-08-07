from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable

from embodied_runtime import evaluation
from embodied_runtime.evaluation import reporting
from embodied_runtime.evaluation.reporting import (
    EvaluationReport,
    ExperimentCondition,
    OracleGateConfig,
    TrialRecord,
)


def test_public_api_is_available_from_package_and_evaluation_namespace() -> None:
    assert evaluation.EvaluationReport is reporting.EvaluationReport
    assert evaluation.ExperimentCondition is reporting.ExperimentCondition
    assert evaluation.OracleGateConfig is reporting.OracleGateConfig
    assert evaluation.TrialRecord is reporting.TrialRecord


def test_report_json_and_csv_are_deterministic_and_machine_readable(
    trial_factory: Callable[..., TrialRecord],
) -> None:
    records = [
        trial_factory(
            2,
            ExperimentCondition.ORACLE_PLAN,
            True,
            edge=(2.0,),
            planner=(8.0,),
        ),
        trial_factory(1, ExperimentCondition.EDGE_ONLY, False, edge=(1.0,)),
        trial_factory(2, ExperimentCondition.EDGE_ONLY, False, edge=(3.0,)),
        trial_factory(
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
