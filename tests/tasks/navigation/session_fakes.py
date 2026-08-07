from __future__ import annotations

import asyncio

from embodied_runtime.tasks.navigation import (
    EncodedRGBFrame,
    MobileBaseState,
    NavigationObservation,
    NavigationRequest,
    PlanarVelocityCommand,
    WaypointPlan,
)
from embodied_runtime.utils import Pose2D

_JPEG = b"\xff\xd8\xff\xd9"


class ObservationSource:
    def __init__(self) -> None:
        self.sequence = 0
        self.resets: list[bool] = []

    def capture(self, *, episode_id, reset, robot_state):
        self.sequence += 1
        self.resets.append(reset)
        frame = EncodedRGBFrame(
            self.sequence,
            100.0 + self.sequence * 0.01,
            _JPEG,
            width=1,
            height=1,
        )
        return NavigationObservation(
            episode_id,
            self.sequence,
            reset,
            (frame,),
            robot_state=robot_state,
        )


class MobileBase:
    def __init__(self, *, preflight_error: Exception | None = None) -> None:
        self.preflight_error = preflight_error
        self.preflight_calls = 0
        self.state_calls = 0
        self.started: list[tuple[PlanarVelocityCommand, float]] = []
        self.updated: list[PlanarVelocityCommand] = []
        self.stop_calls = 0

    def _state(self) -> MobileBaseState:
        return MobileBaseState(Pose2D(0, 0, 0), 0, 0, 0, self.state_calls, 100.0)

    def preflight(self) -> MobileBaseState:
        self.preflight_calls += 1
        if self.preflight_error is not None:
            raise self.preflight_error
        return self._state()

    def state(self) -> MobileBaseState:
        self.state_calls += 1
        return self._state()

    def start_velocity_lease(self, command, *, duration_s):
        self.started.append((command, duration_s))
        return "lease-1"

    def update_velocity_lease(self, lease_id, command):
        assert lease_id == "lease-1"
        self.updated.append(command)

    def stop(self):
        self.stop_calls += 1


class Policy:
    def __init__(self, plans: list[WaypointPlan], delays: tuple[float, ...] = ()) -> None:
        self.plans = plans
        self.delays = delays
        self.requests: list[NavigationRequest] = []

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index < len(self.delays):
            await asyncio.sleep(self.delays[index])
        return self.plans[index]
