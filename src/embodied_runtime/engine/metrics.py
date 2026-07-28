"""Small dependency-free metrics collector for the prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    submitted: int
    started: int
    succeeded: int
    failed: int
    cancelled: int
    deadline_exceeded: int
    rejected: int
    batches: int
    stage_calls: int
    queue_depth: int
    peak_queue_depth: int
    total_queue_time_s: float
    total_execution_time_s: float
    stage_time_s: dict[str, float] = field(default_factory=dict)


class EngineMetrics:
    """Thread-safe counters intentionally decoupled from a metrics vendor."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._submitted = 0
        self._started = 0
        self._succeeded = 0
        self._failed = 0
        self._cancelled = 0
        self._deadline_exceeded = 0
        self._rejected = 0
        self._batches = 0
        self._stage_calls = 0
        self._queue_depth = 0
        self._peak_queue_depth = 0
        self._total_queue_time_s = 0.0
        self._total_execution_time_s = 0.0
        self._stage_time_s: dict[str, float] = {}

    def submitted(self, queue_depth: int) -> None:
        with self._lock:
            self._submitted += 1
            self._queue_depth = queue_depth
            self._peak_queue_depth = max(self._peak_queue_depth, queue_depth)

    def queue_depth(self, value: int) -> None:
        with self._lock:
            self._queue_depth = max(0, value)
            self._peak_queue_depth = max(self._peak_queue_depth, self._queue_depth)

    def rejected(self) -> None:
        with self._lock:
            self._rejected += 1

    def batch_started(self, size: int, queue_times_s: list[float]) -> None:
        with self._lock:
            self._batches += 1
            self._started += size
            self._total_queue_time_s += sum(queue_times_s)

    def stage_finished(self, name: str, elapsed_s: float) -> None:
        with self._lock:
            self._stage_calls += 1
            self._stage_time_s[name] = self._stage_time_s.get(name, 0.0) + elapsed_s

    def succeeded(self, count: int, execution_time_s: float) -> None:
        with self._lock:
            self._succeeded += count
            self._total_execution_time_s += execution_time_s * count

    def failed(self, count: int = 1) -> None:
        with self._lock:
            self._failed += count

    def cancelled(self, count: int = 1) -> None:
        with self._lock:
            self._cancelled += count

    def deadline_exceeded(self, count: int = 1) -> None:
        with self._lock:
            self._deadline_exceeded += count

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                submitted=self._submitted,
                started=self._started,
                succeeded=self._succeeded,
                failed=self._failed,
                cancelled=self._cancelled,
                deadline_exceeded=self._deadline_exceeded,
                rejected=self._rejected,
                batches=self._batches,
                stage_calls=self._stage_calls,
                queue_depth=self._queue_depth,
                peak_queue_depth=self._peak_queue_depth,
                total_queue_time_s=self._total_queue_time_s,
                total_execution_time_s=self._total_execution_time_s,
                stage_time_s=dict(self._stage_time_s),
            )
