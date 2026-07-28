"""Minimal boundary for future robot adapters (Group 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .types import Metadata, TensorTree


@dataclass(slots=True)
class RobotObservation:
    timestamp_s: float
    values: TensorTree
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class RobotAction:
    timestamp_s: float
    values: TensorTree
    metadata: Metadata = field(default_factory=dict)


@runtime_checkable
class RobotAdapter(Protocol):
    robot_id: str

    def observe(self) -> RobotObservation: ...

    def execute(self, action: RobotAction) -> None: ...

    def stop(self) -> None: ...
