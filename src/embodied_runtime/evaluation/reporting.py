"""Dependency-free records and statistics for paired embodied-agent experiments.

The module deliberately keeps task execution out of the evaluation layer.  A runner
only needs to emit :class:`TrialRecord` values with a shared ``pair_id`` for trials
that used the same task, seed, initial state, and disturbance schedule.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExperimentCondition(str, Enum):
    """Supported conditions in the big-brain/small-brain comparison."""

    EDGE_ONLY = "edge_only"
    ORACLE_PLAN = "oracle_plan"
    WRONG_PLAN = "wrong_plan"
    OPEN_LOOP = "open_loop"
    CLOSED_LOOP = "closed_loop"


_CONDITION_ORDER = {condition: index for index, condition in enumerate(ExperimentCondition)}
_WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One completed trial.

    ``pair_id`` identifies the controlled scenario shared across conditions.  The
    same ID must therefore have the same ``task_id`` and ``seed`` everywhere in an
    :class:`EvaluationReport`.
    """

    pair_id: str
    task_id: str
    seed: int
    condition: ExperimentCondition
    success: bool
    steps: int
    edge_latencies_ms: tuple[float, ...] = ()
    planner_latencies_ms: tuple[float, ...] = ()
    chunk_generation_latencies_ms: tuple[float, ...] = ()
    queue_hit_latencies_ms: tuple[float, ...] = ()
    deadline_misses: int = 0
    subgoals_completed: int = 0
    subgoals_total: int = 0
    initial_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", _non_empty_text("pair_id", self.pair_id))
        object.__setattr__(self, "task_id", _non_empty_text("task_id", self.task_id))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        try:
            condition = ExperimentCondition(self.condition)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported experiment condition: {self.condition!r}") from error
        object.__setattr__(self, "condition", condition)
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        _non_negative_integer("steps", self.steps)
        edge_latencies = _latencies("edge_latencies_ms", self.edge_latencies_ms)
        planner_latencies = _latencies("planner_latencies_ms", self.planner_latencies_ms)
        chunk_latencies = _latencies(
            "chunk_generation_latencies_ms",
            self.chunk_generation_latencies_ms,
        )
        queue_hit_latencies = _latencies(
            "queue_hit_latencies_ms",
            self.queue_hit_latencies_ms,
        )
        object.__setattr__(self, "edge_latencies_ms", edge_latencies)
        object.__setattr__(self, "planner_latencies_ms", planner_latencies)
        object.__setattr__(self, "chunk_generation_latencies_ms", chunk_latencies)
        object.__setattr__(self, "queue_hit_latencies_ms", queue_hit_latencies)
        _non_negative_integer("deadline_misses", self.deadline_misses)
        if self.deadline_misses > len(edge_latencies):
            raise ValueError("deadline_misses cannot exceed the number of edge latency samples")
        _non_negative_integer("subgoals_completed", self.subgoals_completed)
        _non_negative_integer("subgoals_total", self.subgoals_total)
        if self.subgoals_completed > self.subgoals_total:
            raise ValueError("subgoals_completed cannot exceed subgoals_total")
        if self.initial_fingerprint is not None:
            object.__setattr__(
                self,
                "initial_fingerprint",
                _non_empty_text("initial_fingerprint", self.initial_fingerprint),
            )


@dataclass(frozen=True, slots=True)
class ConditionSummary:
    """Aggregate metrics for one condition."""

    condition: ExperimentCondition
    trials: int
    successes: int
    success_rate: float
    success_rate_ci95_low: float
    success_rate_ci95_high: float
    steps_total: int
    steps_mean: float
    steps_p50: float
    steps_p95: float
    steps_p99: float
    edge_calls: int
    edge_latency_ms_mean: float | None
    edge_latency_ms_p50: float | None
    edge_latency_ms_p95: float | None
    edge_latency_ms_p99: float | None
    planner_calls: int
    planner_latency_ms_mean: float | None
    planner_latency_ms_p50: float | None
    planner_latency_ms_p95: float | None
    planner_latency_ms_p99: float | None
    chunk_generations: int
    chunk_generation_latency_ms_mean: float | None
    chunk_generation_latency_ms_p50: float | None
    chunk_generation_latency_ms_p95: float | None
    chunk_generation_latency_ms_p99: float | None
    queue_hits: int
    queue_hit_latency_ms_mean: float | None
    queue_hit_latency_ms_p50: float | None
    queue_hit_latency_ms_p95: float | None
    queue_hit_latency_ms_p99: float | None
    deadline_misses: int
    deadline_miss_rate: float | None
    subgoals_completed: int
    subgoals_total: int
    subgoal_completion_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition.value,
            "trials": self.trials,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "success_rate_ci95_low": self.success_rate_ci95_low,
            "success_rate_ci95_high": self.success_rate_ci95_high,
            "steps_total": self.steps_total,
            "steps_mean": self.steps_mean,
            "steps_p50": self.steps_p50,
            "steps_p95": self.steps_p95,
            "steps_p99": self.steps_p99,
            "edge_calls": self.edge_calls,
            "edge_latency_ms_mean": self.edge_latency_ms_mean,
            "edge_latency_ms_p50": self.edge_latency_ms_p50,
            "edge_latency_ms_p95": self.edge_latency_ms_p95,
            "edge_latency_ms_p99": self.edge_latency_ms_p99,
            "planner_calls": self.planner_calls,
            "planner_latency_ms_mean": self.planner_latency_ms_mean,
            "planner_latency_ms_p50": self.planner_latency_ms_p50,
            "planner_latency_ms_p95": self.planner_latency_ms_p95,
            "planner_latency_ms_p99": self.planner_latency_ms_p99,
            "chunk_generations": self.chunk_generations,
            "chunk_generation_latency_ms_mean": self.chunk_generation_latency_ms_mean,
            "chunk_generation_latency_ms_p50": self.chunk_generation_latency_ms_p50,
            "chunk_generation_latency_ms_p95": self.chunk_generation_latency_ms_p95,
            "chunk_generation_latency_ms_p99": self.chunk_generation_latency_ms_p99,
            "queue_hits": self.queue_hits,
            "queue_hit_latency_ms_mean": self.queue_hit_latency_ms_mean,
            "queue_hit_latency_ms_p50": self.queue_hit_latency_ms_p50,
            "queue_hit_latency_ms_p95": self.queue_hit_latency_ms_p95,
            "queue_hit_latency_ms_p99": self.queue_hit_latency_ms_p99,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": self.deadline_miss_rate,
            "subgoals_completed": self.subgoals_completed,
            "subgoals_total": self.subgoals_total,
            "subgoal_completion_rate": self.subgoal_completion_rate,
        }


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


@dataclass(frozen=True, slots=True)
class OracleGateConfig:
    """Evidence required before spending resources on learned-planner experiments."""

    minimum_gain_percentage_points: float = 10.0
    minimum_paired_trials: int = 20
    minimum_discordant_pairs: int = 1
    maximum_p_value: float = 0.05
    require_complete_pairs: bool = True

    def __post_init__(self) -> None:
        gain = _finite_number(
            "minimum_gain_percentage_points",
            self.minimum_gain_percentage_points,
        )
        if not 0.0 <= gain <= 100.0:
            raise ValueError("minimum_gain_percentage_points must be between 0 and 100")
        object.__setattr__(self, "minimum_gain_percentage_points", gain)
        _positive_integer("minimum_paired_trials", self.minimum_paired_trials)
        _positive_integer("minimum_discordant_pairs", self.minimum_discordant_pairs)
        maximum_p_value = _finite_number("maximum_p_value", self.maximum_p_value)
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


class EvaluationReport:
    """Validated collection of paired trials with deterministic renderers."""

    def __init__(self, records: Iterable[TrialRecord]) -> None:
        try:
            normalized = tuple(records)
        except TypeError as error:
            raise TypeError("records must be an iterable of TrialRecord values") from error
        if not normalized:
            raise ValueError("an evaluation report requires at least one trial")
        if any(not isinstance(record, TrialRecord) for record in normalized):
            raise TypeError("records must contain only TrialRecord values")

        identities: dict[str, tuple[str, int]] = {}
        fingerprints: dict[str, str] = {}
        seen: set[tuple[ExperimentCondition, str]] = set()
        for record in normalized:
            key = (record.condition, record.pair_id)
            if key in seen:
                raise ValueError("each condition may contain at most one trial for a given pair_id")
            seen.add(key)
            identity = (record.task_id, record.seed)
            previous = identities.setdefault(record.pair_id, identity)
            if previous != identity:
                raise ValueError(
                    "a pair_id must refer to the same task_id and seed in every condition"
                )
            if record.initial_fingerprint is not None:
                previous_fingerprint = fingerprints.setdefault(
                    record.pair_id,
                    record.initial_fingerprint,
                )
                if previous_fingerprint != record.initial_fingerprint:
                    raise ValueError(
                        "paired conditions must start from the same observation fingerprint"
                    )

        self._records = tuple(sorted(normalized, key=_record_sort_key))

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        return self._records

    def summaries(self) -> tuple[ConditionSummary, ...]:
        present = {record.condition for record in self._records}
        return tuple(
            summarize_condition(record for record in self._records if record.condition is condition)
            for condition in ExperimentCondition
            if condition in present
        )

    def comparisons(
        self,
        *,
        baseline: ExperimentCondition = ExperimentCondition.EDGE_ONLY,
    ) -> tuple[PairedComparison, ...]:
        baseline = _condition(baseline)
        present = {record.condition for record in self._records}
        if baseline not in present:
            return ()
        return tuple(
            paired_binary_comparison(
                self._records,
                baseline=baseline,
                comparator=condition,
            )
            for condition in ExperimentCondition
            if condition is not baseline and condition in present
        )

    def oracle_gate(
        self,
        config: OracleGateConfig | None = None,
    ) -> OracleGateResult:
        return evaluate_oracle_gate(self._records, config=config)

    def to_dict(
        self,
        *,
        oracle_gate_config: OracleGateConfig | None = None,
    ) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": 1,
            "trials": [_trial_dict(record) for record in self._records],
            "condition_summaries": [summary.to_dict() for summary in self.summaries()],
            "paired_comparisons": [comparison.to_dict() for comparison in self.comparisons()],
        }
        if oracle_gate_config is not None:
            document["oracle_gate"] = self.oracle_gate(oracle_gate_config).to_dict()
        return document

    def to_json(
        self,
        *,
        oracle_gate_config: OracleGateConfig | None = None,
        indent: int | None = 2,
    ) -> str:
        if indent is not None and (
            isinstance(indent, bool) or not isinstance(indent, int) or indent < 0
        ):
            raise ValueError("indent must be a non-negative integer or None")
        return (
            json.dumps(
                self.to_dict(oracle_gate_config=oracle_gate_config),
                allow_nan=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )

    def trials_csv(self) -> str:
        header = (
            "pair_id",
            "task_id",
            "seed",
            "condition",
            "success",
            "steps",
            "edge_latencies_ms",
            "planner_latencies_ms",
            "chunk_generation_latencies_ms",
            "queue_hit_latencies_ms",
            "deadline_misses",
            "subgoals_completed",
            "subgoals_total",
            "initial_fingerprint",
        )
        rows = (
            (
                record.pair_id,
                record.task_id,
                record.seed,
                record.condition.value,
                int(record.success),
                record.steps,
                _compact_json(record.edge_latencies_ms),
                _compact_json(record.planner_latencies_ms),
                _compact_json(record.chunk_generation_latencies_ms),
                _compact_json(record.queue_hit_latencies_ms),
                record.deadline_misses,
                record.subgoals_completed,
                record.subgoals_total,
                record.initial_fingerprint or "",
            )
            for record in self._records
        )
        return _render_csv(header, rows)

    def summaries_csv(self) -> str:
        dictionaries = tuple(summary.to_dict() for summary in self.summaries())
        header = tuple(ConditionSummary.__dataclass_fields__)
        return _render_csv(
            header,
            (tuple(_csv_value(document[name]) for name in header) for document in dictionaries),
        )

    def comparisons_csv(
        self,
        *,
        baseline: ExperimentCondition = ExperimentCondition.EDGE_ONLY,
    ) -> str:
        header = (
            "baseline",
            "comparator",
            "matched_pairs",
            "baseline_unmatched_trials",
            "comparator_unmatched_trials",
            "both_succeeded",
            "both_failed",
            "baseline_only_succeeded",
            "comparator_only_succeeded",
            "discordant_pairs",
            "effect_percentage_points",
            "exact_two_sided_p_value",
        )
        dictionaries = tuple(
            comparison.to_dict() for comparison in self.comparisons(baseline=baseline)
        )
        return _render_csv(
            header,
            (tuple(_csv_value(document[name]) for name in header) for document in dictionaries),
        )


def summarize_condition(records: Iterable[TrialRecord]) -> ConditionSummary:
    """Summarize records from exactly one experimental condition."""

    normalized = tuple(records)
    if not normalized:
        raise ValueError("condition summary requires at least one trial")
    if any(not isinstance(record, TrialRecord) for record in normalized):
        raise TypeError("condition records must contain only TrialRecord values")
    conditions = {record.condition for record in normalized}
    if len(conditions) != 1:
        raise ValueError("condition summary cannot mix experiment conditions")
    condition = next(iter(conditions))
    successes = sum(record.success for record in normalized)
    success_rate = successes / len(normalized)
    ci_low, ci_high = _wilson_interval(successes, len(normalized))
    steps = tuple(float(record.steps) for record in normalized)
    edge_latencies = tuple(latency for record in normalized for latency in record.edge_latencies_ms)
    planner_latencies = tuple(
        latency for record in normalized for latency in record.planner_latencies_ms
    )
    chunk_latencies = tuple(
        latency for record in normalized for latency in record.chunk_generation_latencies_ms
    )
    queue_hit_latencies = tuple(
        latency for record in normalized for latency in record.queue_hit_latencies_ms
    )
    deadline_misses = sum(record.deadline_misses for record in normalized)
    subgoals_completed = sum(record.subgoals_completed for record in normalized)
    subgoals_total = sum(record.subgoals_total for record in normalized)
    return ConditionSummary(
        condition=condition,
        trials=len(normalized),
        successes=successes,
        success_rate=success_rate,
        success_rate_ci95_low=ci_low,
        success_rate_ci95_high=ci_high,
        steps_total=sum(record.steps for record in normalized),
        steps_mean=sum(steps) / len(steps),
        steps_p50=_percentile(steps, 50),
        steps_p95=_percentile(steps, 95),
        steps_p99=_percentile(steps, 99),
        edge_calls=len(edge_latencies),
        edge_latency_ms_mean=_mean_or_none(edge_latencies),
        edge_latency_ms_p50=_percentile_or_none(edge_latencies, 50),
        edge_latency_ms_p95=_percentile_or_none(edge_latencies, 95),
        edge_latency_ms_p99=_percentile_or_none(edge_latencies, 99),
        planner_calls=len(planner_latencies),
        planner_latency_ms_mean=_mean_or_none(planner_latencies),
        planner_latency_ms_p50=_percentile_or_none(planner_latencies, 50),
        planner_latency_ms_p95=_percentile_or_none(planner_latencies, 95),
        planner_latency_ms_p99=_percentile_or_none(planner_latencies, 99),
        chunk_generations=len(chunk_latencies),
        chunk_generation_latency_ms_mean=_mean_or_none(chunk_latencies),
        chunk_generation_latency_ms_p50=_percentile_or_none(chunk_latencies, 50),
        chunk_generation_latency_ms_p95=_percentile_or_none(chunk_latencies, 95),
        chunk_generation_latency_ms_p99=_percentile_or_none(chunk_latencies, 99),
        queue_hits=len(queue_hit_latencies),
        queue_hit_latency_ms_mean=_mean_or_none(queue_hit_latencies),
        queue_hit_latency_ms_p50=_percentile_or_none(queue_hit_latencies, 50),
        queue_hit_latency_ms_p95=_percentile_or_none(queue_hit_latencies, 95),
        queue_hit_latency_ms_p99=_percentile_or_none(queue_hit_latencies, 99),
        deadline_misses=deadline_misses,
        deadline_miss_rate=(deadline_misses / len(edge_latencies) if edge_latencies else None),
        subgoals_completed=subgoals_completed,
        subgoals_total=subgoals_total,
        subgoal_completion_rate=(subgoals_completed / subgoals_total if subgoals_total else None),
    )


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

    baseline = _condition(baseline)
    comparator = _condition(comparator)
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


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > 256:
        raise ValueError(f"{name} must be at most 256 characters")
    return normalized


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _positive_integer(name: str, value: int) -> None:
    _non_negative_integer(name, value)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")


def _latencies(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    normalized = tuple(_finite_number(name, value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{name} values must be non-negative")
    return normalized


def _condition(value: ExperimentCondition) -> ExperimentCondition:
    try:
        return ExperimentCondition(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported experiment condition: {value!r}") from error


def _record_sort_key(record: TrialRecord) -> tuple[int, str, int, str]:
    return (
        _CONDITION_ORDER[record.condition],
        record.task_id,
        record.seed,
        record.pair_id,
    )


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    proportion = successes / trials
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        _WILSON_Z_95
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile over a non-empty sequence."""

    if not values:
        raise ValueError("a percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _percentile_or_none(
    values: Sequence[float],
    percentile: float,
) -> float | None:
    return _percentile(values, percentile) if values else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


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


def _trial_dict(record: TrialRecord) -> dict[str, Any]:
    return {
        "pair_id": record.pair_id,
        "task_id": record.task_id,
        "seed": record.seed,
        "condition": record.condition.value,
        "success": record.success,
        "steps": record.steps,
        "edge_latencies_ms": list(record.edge_latencies_ms),
        "planner_latencies_ms": list(record.planner_latencies_ms),
        "chunk_generation_latencies_ms": list(record.chunk_generation_latencies_ms),
        "queue_hit_latencies_ms": list(record.queue_hit_latencies_ms),
        "deadline_misses": record.deadline_misses,
        "subgoals_completed": record.subgoals_completed,
        "subgoals_total": record.subgoals_total,
        "initial_fingerprint": record.initial_fingerprint,
    }


def _compact_json(values: Sequence[float]) -> str:
    return json.dumps(list(values), allow_nan=False, separators=(",", ":"))


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _render_csv(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()
