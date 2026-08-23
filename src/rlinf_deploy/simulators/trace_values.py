"""Immutable event values for deterministic simulator episode traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from rlinf_deploy.robots.action import RobotAction
from rlinf_deploy.robots.observation import RobotObservation
from rlinf_deploy.types import Metadata

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


def _validate_sequence(sequence: int) -> None:
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("trace event sequence must be a non-negative integer")


__all__ = [
    "DisturbanceTraceEvent",
    "ResetTraceEvent",
    "StepTraceEvent",
    "TraceEvent",
]
