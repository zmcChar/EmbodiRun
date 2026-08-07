"""Validated records for individual evaluation trials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._validation import latencies, non_empty_text, non_negative_integer


class ExperimentCondition(str, Enum):
    """Supported conditions in the big-brain/small-brain comparison."""

    EDGE_ONLY = "edge_only"
    ORACLE_PLAN = "oracle_plan"
    WRONG_PLAN = "wrong_plan"
    OPEN_LOOP = "open_loop"
    CLOSED_LOOP = "closed_loop"


_CONDITION_ORDER = {condition: index for index, condition in enumerate(ExperimentCondition)}


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One completed trial.

    ``pair_id`` identifies the controlled scenario shared across conditions. The
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
        object.__setattr__(self, "pair_id", non_empty_text("pair_id", self.pair_id))
        object.__setattr__(self, "task_id", non_empty_text("task_id", self.task_id))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "condition", normalize_condition(self.condition))
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        non_negative_integer("steps", self.steps)
        edge_latencies = latencies("edge_latencies_ms", self.edge_latencies_ms)
        planner_latencies = latencies("planner_latencies_ms", self.planner_latencies_ms)
        chunk_latencies = latencies(
            "chunk_generation_latencies_ms",
            self.chunk_generation_latencies_ms,
        )
        queue_hit_latencies = latencies(
            "queue_hit_latencies_ms",
            self.queue_hit_latencies_ms,
        )
        object.__setattr__(self, "edge_latencies_ms", edge_latencies)
        object.__setattr__(self, "planner_latencies_ms", planner_latencies)
        object.__setattr__(self, "chunk_generation_latencies_ms", chunk_latencies)
        object.__setattr__(self, "queue_hit_latencies_ms", queue_hit_latencies)
        non_negative_integer("deadline_misses", self.deadline_misses)
        if self.deadline_misses > len(edge_latencies):
            raise ValueError("deadline_misses cannot exceed the number of edge latency samples")
        non_negative_integer("subgoals_completed", self.subgoals_completed)
        non_negative_integer("subgoals_total", self.subgoals_total)
        if self.subgoals_completed > self.subgoals_total:
            raise ValueError("subgoals_completed cannot exceed subgoals_total")
        if self.initial_fingerprint is not None:
            object.__setattr__(
                self,
                "initial_fingerprint",
                non_empty_text("initial_fingerprint", self.initial_fingerprint),
            )


def normalize_condition(value: ExperimentCondition) -> ExperimentCondition:
    try:
        return ExperimentCondition(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported experiment condition: {value!r}") from error


def record_sort_key(record: TrialRecord) -> tuple[int, str, int, str]:
    return (
        _CONDITION_ORDER[record.condition],
        record.task_id,
        record.seed,
        record.pair_id,
    )
