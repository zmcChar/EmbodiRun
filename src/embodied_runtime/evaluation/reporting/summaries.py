"""Aggregate metrics for a single evaluation condition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ._statistics import mean_or_none, percentile, percentile_or_none, wilson_interval
from .records import ExperimentCondition, TrialRecord


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
    ci_low, ci_high = wilson_interval(successes, len(normalized))
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
        steps_p50=percentile(steps, 50),
        steps_p95=percentile(steps, 95),
        steps_p99=percentile(steps, 99),
        edge_calls=len(edge_latencies),
        edge_latency_ms_mean=mean_or_none(edge_latencies),
        edge_latency_ms_p50=percentile_or_none(edge_latencies, 50),
        edge_latency_ms_p95=percentile_or_none(edge_latencies, 95),
        edge_latency_ms_p99=percentile_or_none(edge_latencies, 99),
        planner_calls=len(planner_latencies),
        planner_latency_ms_mean=mean_or_none(planner_latencies),
        planner_latency_ms_p50=percentile_or_none(planner_latencies, 50),
        planner_latency_ms_p95=percentile_or_none(planner_latencies, 95),
        planner_latency_ms_p99=percentile_or_none(planner_latencies, 99),
        chunk_generations=len(chunk_latencies),
        chunk_generation_latency_ms_mean=mean_or_none(chunk_latencies),
        chunk_generation_latency_ms_p50=percentile_or_none(chunk_latencies, 50),
        chunk_generation_latency_ms_p95=percentile_or_none(chunk_latencies, 95),
        chunk_generation_latency_ms_p99=percentile_or_none(chunk_latencies, 99),
        queue_hits=len(queue_hit_latencies),
        queue_hit_latency_ms_mean=mean_or_none(queue_hit_latencies),
        queue_hit_latency_ms_p50=percentile_or_none(queue_hit_latencies, 50),
        queue_hit_latency_ms_p95=percentile_or_none(queue_hit_latencies, 95),
        queue_hit_latency_ms_p99=percentile_or_none(queue_hit_latencies, 99),
        deadline_misses=deadline_misses,
        deadline_miss_rate=(deadline_misses / len(edge_latencies) if edge_latencies else None),
        subgoals_completed=subgoals_completed,
        subgoals_total=subgoals_total,
        subgoal_completion_rate=(subgoals_completed / subgoals_total if subgoals_total else None),
    )
