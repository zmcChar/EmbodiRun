"""Dependency-free records and statistics for paired embodied-agent experiments.

The package deliberately keeps task execution out of the evaluation layer. A runner
only needs to emit :class:`TrialRecord` values with a shared ``pair_id`` for trials
that used the same task, seed, initial state, and disturbance schedule.
"""

from .comparisons import PairedComparison, paired_binary_comparison
from .oracle import OracleGateConfig, OracleGateResult, evaluate_oracle_gate
from .records import ExperimentCondition, TrialRecord
from .report import EvaluationReport
from .summaries import ConditionSummary, summarize_condition

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
