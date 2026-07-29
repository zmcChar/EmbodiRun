"""Deterministic, dependency-free episode traces for simulator evaluation."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from embodied_runtime.contracts import Metadata, RobotAction, RobotObservation

from .base import EpisodeStep


@dataclass(frozen=True, slots=True)
class ResetTraceEvent:
    sequence: int
    observation: RobotObservation
    kind: Literal["reset"] = field(default="reset", init=False)

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("reset trace observation must be a RobotObservation")


@dataclass(frozen=True, slots=True)
class StepTraceEvent:
    sequence: int
    action: RobotAction
    outcome: EpisodeStep
    kind: Literal["step"] = field(default="step", init=False)

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        if not isinstance(self.action, RobotAction):
            raise TypeError("step trace action must be a RobotAction")
        if not isinstance(self.outcome, EpisodeStep):
            raise TypeError("step trace outcome must be an EpisodeStep")


@dataclass(frozen=True, slots=True)
class DisturbanceTraceEvent:
    sequence: int
    name: str
    details: Metadata = field(default_factory=dict)
    kind: Literal["disturbance"] = field(default="disturbance", init=False)

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("disturbance name must be a non-empty string")
        if not isinstance(self.details, Mapping):
            raise TypeError("disturbance details must be a mapping")
        if any(not isinstance(key, str) for key in self.details):
            raise TypeError("disturbance detail keys must be strings")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "details", dict(self.details))


TraceEvent: TypeAlias = ResetTraceEvent | StepTraceEvent | DisturbanceTraceEvent


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Immutable snapshot of one ordered episode event stream."""

    task: str | None
    seed: int | None
    events: tuple[TraceEvent, ...]

    def __post_init__(self) -> None:
        task, seed = _validate_identity(self.task, self.seed)
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
        for index, event in enumerate(events[:-1]):
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

        steps = self.steps
        final_outcome = steps[-1].outcome if steps else None
        return {
            "task": self.task,
            "seed": self.seed,
            "event_count": len(self.events),
            "step_count": len(steps),
            "disturbance_count": len(self.disturbances),
            "total_reward": sum(step.outcome.reward for step in steps),
            "terminated": final_outcome.terminated if final_outcome is not None else False,
            "truncated": final_outcome.truncated if final_outcome is not None else False,
            "success": final_outcome.success if final_outcome is not None else None,
            "events": [_event_summary(event) for event in self.events],
        }


class TraceRecorder:
    """Record one episode with stable, zero-based event ordering."""

    def __init__(self, *, task: str | None = None, seed: int | None = None) -> None:
        self._task, self._seed = _validate_identity(task, seed)
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


def _validate_identity(task: str | None, seed: int | None) -> tuple[str | None, int | None]:
    if task is not None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("trace task must be a non-empty string or None")
        task = task.strip()
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("trace seed must be an int or None")
    return task, seed


def _validate_sequence(sequence: int) -> None:
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("trace event sequence must be a non-negative integer")


def _snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception as error:
        raise TypeError("trace values must support deterministic snapshots") from error


def _event_summary(event: TraceEvent) -> dict[str, Any]:
    if isinstance(event, ResetTraceEvent):
        return {
            "sequence": event.sequence,
            "kind": event.kind,
            "observation": _observation_summary(event.observation),
        }
    if isinstance(event, StepTraceEvent):
        return {
            "sequence": event.sequence,
            "kind": event.kind,
            "action": _action_summary(event.action),
            "outcome": {
                "observation": _observation_summary(event.outcome.observation),
                "reward": event.outcome.reward,
                "terminated": event.outcome.terminated,
                "truncated": event.outcome.truncated,
                "success": event.outcome.success,
                "subgoal_progress": event.outcome.subgoal_progress,
                "info": _json_compatible(event.outcome.info),
            },
        }
    return {
        "sequence": event.sequence,
        "kind": event.kind,
        "name": event.name,
        "details": _json_compatible(event.details),
    }


def _observation_summary(observation: RobotObservation) -> dict[str, Any]:
    return {
        "timestamp_s": _finite_json_float(observation.timestamp_s),
        "values": _json_compatible(observation.values),
        "metadata": _json_compatible(observation.metadata),
    }


def _action_summary(action: RobotAction) -> dict[str, Any]:
    return {
        "timestamp_s": _finite_json_float(action.timestamp_s),
        "values": _json_compatible(action.values),
        "metadata": _json_compatible(action.metadata),
    }


def _finite_json_float(value: Any) -> float | str:
    normalized = float(value)
    if math.isfinite(normalized):
        return normalized
    if math.isnan(normalized):
        return "NaN"
    return "Infinity" if normalized > 0 else "-Infinity"


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_json_float(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            json_key = str(key)
            if json_key in converted:
                raise ValueError("trace mapping keys collide after JSON conversion")
            converted[json_key] = _json_compatible(value[key])
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted_items = [_json_compatible(item) for item in value]
        return sorted(
            converted_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_compatible(tolist())
    value_type = type(value)
    return {"type": f"{value_type.__module__}.{value_type.__qualname__}"}


__all__ = [
    "DisturbanceTraceEvent",
    "EpisodeTrace",
    "ResetTraceEvent",
    "StepTraceEvent",
    "TraceEvent",
    "TraceRecorder",
]
