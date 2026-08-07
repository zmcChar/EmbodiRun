"""Evidence gate for enabling learned-planner experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ._validation import finite_number, positive_integer
from .comparisons import PairedComparison, paired_binary_comparison
from .records import ExperimentCondition, TrialRecord


@dataclass(frozen=True, slots=True)
class OracleGateConfig:
    """Evidence required before spending resources on learned-planner experiments."""

    minimum_gain_percentage_points: float = 10.0
    minimum_paired_trials: int = 20
    minimum_discordant_pairs: int = 1
    maximum_p_value: float = 0.05
    require_complete_pairs: bool = True

    def __post_init__(self) -> None:
        gain = finite_number(
            "minimum_gain_percentage_points",
            self.minimum_gain_percentage_points,
        )
        if not 0.0 <= gain <= 100.0:
            raise ValueError("minimum_gain_percentage_points must be between 0 and 100")
        object.__setattr__(self, "minimum_gain_percentage_points", gain)
        positive_integer("minimum_paired_trials", self.minimum_paired_trials)
        positive_integer("minimum_discordant_pairs", self.minimum_discordant_pairs)
        maximum_p_value = finite_number("maximum_p_value", self.maximum_p_value)
        if not 0.0 < maximum_p_value <= 1.0:
            raise ValueError("maximum_p_value must be greater than 0 and at most 1")
        object.__setattr__(self, "maximum_p_value", maximum_p_value)
        if not isinstance(self.require_complete_pairs, bool):
            raise TypeError("require_complete_pairs must be a bool")


@dataclass(frozen=True, slots=True)
class OracleGateResult:
    """Auditable result of applying an :class:`OracleGateConfig`."""

    passed: bool
    comparison: PairedComparison
    config: OracleGateConfig
    failed_checks: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed_checks": list(self.failed_checks),
            "config": {
                "minimum_gain_percentage_points": (self.config.minimum_gain_percentage_points),
                "minimum_paired_trials": self.config.minimum_paired_trials,
                "minimum_discordant_pairs": self.config.minimum_discordant_pairs,
                "maximum_p_value": self.config.maximum_p_value,
                "require_complete_pairs": self.config.require_complete_pairs,
            },
            "comparison": self.comparison.to_dict(),
        }


def evaluate_oracle_gate(
    records: Iterable[TrialRecord],
    *,
    config: OracleGateConfig | None = None,
) -> OracleGateResult:
    """Require paired Oracle evidence before enabling learned planning."""

    normalized = tuple(records)
    config = config or OracleGateConfig()
    if not isinstance(config, OracleGateConfig):
        raise TypeError("config must be an OracleGateConfig or None")
    comparison = paired_binary_comparison(
        normalized,
        baseline=ExperimentCondition.EDGE_ONLY,
        comparator=ExperimentCondition.ORACLE_PLAN,
    )
    failed: list[str] = []
    if comparison.matched_pairs < config.minimum_paired_trials:
        failed.append("insufficient_paired_trials")
    if comparison.discordant_pairs < config.minimum_discordant_pairs:
        failed.append("insufficient_discordant_pairs")
    if comparison.effect_percentage_points < config.minimum_gain_percentage_points:
        failed.append("minimum_gain_not_met")
    if comparison.exact_two_sided_p_value > config.maximum_p_value:
        failed.append("paired_significance_not_met")
    if config.require_complete_pairs and (
        comparison.baseline_unmatched_trials or comparison.comparator_unmatched_trials
    ):
        failed.append("incomplete_pairing")
    return OracleGateResult(
        passed=not failed,
        comparison=comparison,
        config=config,
        failed_checks=tuple(failed),
    )
