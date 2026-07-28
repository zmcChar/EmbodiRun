from __future__ import annotations

from typing import Protocol, runtime_checkable

from embodied_runtime.contracts import RawRequest, RobotObservation

from .profile import RobotProfile


@runtime_checkable
class ObservationMapper(Protocol):
    """Convert one robot's sensor schema into the model-facing request."""

    def map_observation(
        self,
        observation: RobotObservation,
        profile: RobotProfile,
    ) -> RawRequest: ...
