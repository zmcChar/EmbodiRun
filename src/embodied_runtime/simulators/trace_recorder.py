"""Mutable recorder for immutable deterministic episode traces."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from embodied_runtime.robots.action import RobotAction
from embodied_runtime.robots.observation import RobotObservation

from .base import EpisodeStep
from .trace import EpisodeTrace, validate_trace_identity
from .trace_values import (
    DisturbanceTraceEvent,
    ResetTraceEvent,
    StepTraceEvent,
    TraceEvent,
)


class TraceRecorder:
    """Record one episode with stable, zero-based event ordering."""

    def __init__(self, *, task: str | None = None, seed: int | None = None) -> None:
        self._task, self._seed = validate_trace_identity(task, seed)
        self._events: list[TraceEvent] = []
        self._terminal = False

    def record_reset(self, observation: RobotObservation) -> None:
        if self._events:
            raise RuntimeError("trace recorder already contains a reset event")
        if not isinstance(observation, RobotObservation):
            raise TypeError("reset trace observation must be a RobotObservation")
        self._events.append(ResetTraceEvent(sequence=0, observation=_snapshot(observation)))

    def record_step(self, action: RobotAction, outcome: EpisodeStep) -> None:
        self._require_active()
        if not isinstance(action, RobotAction):
            raise TypeError("step trace action must be a RobotAction")
        if not isinstance(outcome, EpisodeStep):
            raise TypeError("step trace outcome must be an EpisodeStep")
        self._events.append(
            StepTraceEvent(
                sequence=len(self._events),
                action=_snapshot(action),
                outcome=_snapshot(outcome),
            )
        )
        self._terminal = outcome.done

    def record_disturbance(
        self,
        name: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_active()
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("disturbance details must be a mapping or None")
        self._events.append(
            DisturbanceTraceEvent(
                sequence=len(self._events),
                name=name,
                details=_snapshot(dict(details or {})),
            )
        )

    def snapshot(self) -> EpisodeTrace:
        if not self._events:
            raise RuntimeError("trace recorder has not recorded a reset")
        return EpisodeTrace(
            task=self._task,
            seed=self._seed,
            events=_snapshot(tuple(self._events)),
        )

    def summary(self) -> dict[str, Any]:
        return self.snapshot().summary()

    def _require_active(self) -> None:
        if not self._events:
            raise RuntimeError("trace recorder must record reset before episode events")
        if self._terminal:
            raise RuntimeError("trace recorder cannot append events after a terminal step")


def _snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception as error:
        raise TypeError("trace values must support deterministic snapshots") from error


__all__ = ["TraceRecorder"]
