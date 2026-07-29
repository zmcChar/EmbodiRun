"""Scientific reporting for paired embodied-runtime experiments."""

from .reporting import (
    ConditionSummary,
    EvaluationReport,
    ExperimentCondition,
    OracleGateConfig,
    OracleGateResult,
    PairedComparison,
    TrialRecord,
    evaluate_oracle_gate,
    paired_binary_comparison,
    summarize_condition,
)

__all__ = [
    "ConditionSummary",
    "EvaluationReport",
    "ExperimentCondition",
    "OracleGateConfig",
    "OracleGateResult",
    "PairedComparison",
    "TrialRecord",
    "evaluate_oracle_gate",
    "paired_binary_comparison",
    "summarize_condition",
]
