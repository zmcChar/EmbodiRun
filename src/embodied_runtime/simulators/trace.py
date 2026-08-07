"""Validated aggregate for one deterministic simulator episode trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trace_serialization import summarize_episode_trace
from .trace_values import (
    DisturbanceTraceEvent,
    ResetTraceEvent,
    StepTraceEvent,
    TraceEvent,
)


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Immutable snapshot of one ordered episode event stream."""

    task: str | None
    seed: int | None
    events: tuple[TraceEvent, ...]

    def __post_init__(self) -> None:
        task, seed = validate_trace_identity(self.task, self.seed)
        events = tuple(self.events)
        if not events or not isinstance(events[0], ResetTraceEvent):
            raise ValueError("episode trace must begin with one reset event")
        if any(isinstance(event, ResetTraceEvent) for event in events[1:]):
            raise ValueError("episode trace may contain only one reset event")
        for expected_sequence, event in enumerate(events):
            if not isinstance(
                event,
                (ResetTraceEvent, StepTraceEvent, DisturbanceTraceEvent),
            ):
                raise TypeError("episode trace contains an unsupported event")
            if event.sequence != expected_sequence:
                raise ValueError("episode trace event sequences must be contiguous from zero")
        for event in events[:-1]:
            if isinstance(event, StepTraceEvent) and event.outcome.done:
                raise ValueError("a terminal step must be the final trace event")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "events", events)

    @property
    def steps(self) -> tuple[StepTraceEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, StepTraceEvent))

    @property
    def disturbances(self) -> tuple[DisturbanceTraceEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, DisturbanceTraceEvent))

    def summary(self) -> dict[str, Any]:
        """Return a deterministic structure accepted by strict JSON encoders."""

        return summarize_episode_trace(self)


def validate_trace_identity(
    task: str | None,
    seed: int | None,
) -> tuple[str | None, int | None]:
    if task is not None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("trace task must be a non-empty string or None")
        task = task.strip()
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("trace seed must be an int or None")
    return task, seed


__all__ = [
    "DisturbanceTraceEvent",
    "EpisodeTrace",
    "ResetTraceEvent",
    "StepTraceEvent",
    "TraceEvent",
    "validate_trace_identity",
]
