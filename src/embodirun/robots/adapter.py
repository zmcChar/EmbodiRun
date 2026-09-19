"""Common values and boundary implemented by physical robot adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from embodirun.types import Metadata, TensorTree


@dataclass(slots=True)
class RobotObservation:
    """Timestamped sensor state in one robot's observation schema."""

    timestamp_s: float
    values: TensorTree
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class RobotAction:
    """Timestamped command in one robot's declared action space."""

    timestamp_s: float
    values: TensorTree
    metadata: Metadata = field(default_factory=dict)


class RobotAdapter(ABC):
    """Own one hardware connection and translate Deploy's robot contracts."""

    @abstractmethod
    def connect(self) -> None:
        """Establish the hardware connection without commanding motion."""

    @abstractmethod
    def observe(self) -> RobotObservation:
        """Read SDK state and return a normalized robot observation."""

    @abstractmethod
    def execute(self, action: RobotAction) -> None:
        """Validate and translate a normalized action into an SDK command."""

    @abstractmethod
    def stop(self) -> None:
        """Bring commanded motion to a safe stop or hold state."""

    @abstractmethod
    def close(self) -> None:
        """Release hardware and transport resources."""


__all__ = ["RobotAction", "RobotAdapter", "RobotObservation"]
