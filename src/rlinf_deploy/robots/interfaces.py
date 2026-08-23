"""Minimal lifecycle implemented by physical robot adapters."""

from typing import Protocol, runtime_checkable

from .action import RobotAction
from .observation import RobotObservation


@runtime_checkable
class RobotAdapter(Protocol):
    robot_id: str

    def observe(self) -> RobotObservation: ...

    def execute(self, action: RobotAction) -> None: ...

    def stop(self) -> None: ...


__all__ = ["RobotAdapter"]
