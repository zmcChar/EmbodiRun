"""Exact paired binary comparisons across experiment conditions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .records import ExperimentCondition, TrialRecord, normalize_condition


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Exact paired binary comparison against a baseline condition."""

    baseline: ExperimentCondition
    comparator: ExperimentCondition
    matched_pairs: int
    baseline_unmatched_trials: int
    comparator_unmatched_trials: int
    both_succeeded: int
    both_failed: int
    baseline_only_succeeded: int
    comparator_only_succeeded: int
    effect_percentage_points: float
    exact_two_sided_p_value: float

    @property
    def discordant_pairs(self) -> int:
        return self.baseline_only_succeeded + self.comparator_only_succeeded

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.value,
            "comparator": self.comparator.value,
            "matched_pairs": self.matched_pairs,
            "baseline_unmatched_trials": self.baseline_unmatched_trials,
            "comparator_unmatched_trials": self.comparator_unmatched_trials,
            "both_succeeded": self.both_succeeded,
            "both_failed": self.both_failed,
            "baseline_only_succeeded": self.baseline_only_succeeded,
            "comparator_only_succeeded": self.comparator_only_succeeded,
            "discordant_pairs": self.discordant_pairs,
            "effect_percentage_points": self.effect_percentage_points,
            "exact_two_sided_p_value": self.exact_two_sided_p_value,
        }


def paired_binary_comparison(
    records: Iterable[TrialRecord],
    *,
    baseline: ExperimentCondition = ExperimentCondition.EDGE_ONLY,
    comparator: ExperimentCondition,
) -> PairedComparison:
    """Compare binary success on shared ``pair_id`` values.

    The p-value is the exact two-sided McNemar test, equivalently a two-sided
    binomial test over only discordant pairs under ``p=0.5``.
    """

    baseline = normalize_condition(baseline)
    comparator = normalize_condition(comparator)
    if baseline is comparator:
        raise ValueError("baseline and comparator conditions must differ")
    normalized = tuple(records)
    if any(not isinstance(record, TrialRecord) for record in normalized):
        raise TypeError("records must contain only TrialRecord values")
    baseline_by_pair = _records_by_pair(normalized, baseline)
    comparator_by_pair = _records_by_pair(normalized, comparator)
    shared_pairs = sorted(set(baseline_by_pair) & set(comparator_by_pair))
    if not shared_pairs:
        raise ValueError(f"no paired trials exist for {baseline.value!r} and {comparator.value!r}")

    both_succeeded = 0
    both_failed = 0
    baseline_only_succeeded = 0
    comparator_only_succeeded = 0
    for pair_id in shared_pairs:
        baseline_success = baseline_by_pair[pair_id].success
        comparator_success = comparator_by_pair[pair_id].success
        if baseline_success and comparator_success:
            both_succeeded += 1
        elif not baseline_success and not comparator_success:
            both_failed += 1
        elif baseline_success:
            baseline_only_succeeded += 1
        else:
            comparator_only_succeeded += 1

    matched_pairs = len(shared_pairs)
    effect = (comparator_only_succeeded - baseline_only_succeeded) / matched_pairs * 100.0
    p_value = _exact_two_sided_binomial_p(
        baseline_only_succeeded,
        comparator_only_succeeded,
    )
    return PairedComparison(
        baseline=baseline,
        comparator=comparator,
        matched_pairs=matched_pairs,
        baseline_unmatched_trials=len(set(baseline_by_pair) - set(comparator_by_pair)),
        comparator_unmatched_trials=len(set(comparator_by_pair) - set(baseline_by_pair)),
        both_succeeded=both_succeeded,
        both_failed=both_failed,
        baseline_only_succeeded=baseline_only_succeeded,
        comparator_only_succeeded=comparator_only_succeeded,
        effect_percentage_points=effect,
        exact_two_sided_p_value=p_value,
    )


def _records_by_pair(
    records: Sequence[TrialRecord],
    condition: ExperimentCondition,
) -> dict[str, TrialRecord]:
    selected: dict[str, TrialRecord] = {}
    for record in records:
        if record.condition is not condition:
            continue
        if record.pair_id in selected:
            raise ValueError("each condition may contain at most one trial for a given pair_id")
        selected[record.pair_id] = record
    return selected


def _exact_two_sided_binomial_p(left: int, right: int) -> float:
    discordant = left + right
    if discordant == 0:
        return 1.0
    lower_tail_count = sum(math.comb(discordant, index) for index in range(min(left, right) + 1))
    return min(1.0, 2.0 * lower_tail_count / (2**discordant))
