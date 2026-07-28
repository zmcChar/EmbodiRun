from __future__ import annotations

from typing import Protocol, runtime_checkable

from embodied_runtime.contracts import ActionChunk, RobotAction

from .profile import RobotProfile


@runtime_checkable
class ActionMapper(Protocol):
    """Validate and map a generic model action into a robot command."""

    def map_action(
        self,
        chunk: ActionChunk,
        profile: RobotProfile,
    ) -> RobotAction: ...
