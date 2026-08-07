"""Deterministic JSON and CSV serialization primitives."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

from .records import TrialRecord


def trial_dict(record: TrialRecord) -> dict[str, Any]:
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


def compact_json(values: Sequence[float]) -> str:
    return json.dumps(list(values), allow_nan=False, separators=(",", ":"))


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Enum):
        return value.value
    return value


def render_csv(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()
