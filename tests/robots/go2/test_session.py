from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from embodied_runtime.contracts import (
    EncodedRGBFrame,
    InferenceRequest,
    InferenceResult,
    NavigationObservation,
    NavigationRequest,
    Waypoint,
    WaypointPlan,
)
from embodied_runtime.robots.go2.session import (
    Go2NavigationSession,
    Go2NavigationSessionConfig,
    Go2NavigationSessionError,
)
from embodied_runtime.robots.go2.types import BaseVelocityCommand, Go2State
from embodied_runtime.utils import Pose2D

_JPEG = b"\xff\xd8\xff\xd9"


class _Camera:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, dict[str, object]]] = []
        self.sequence = 0

    def capture(
        self,
        *,
        episode_id: str,
        reset: bool,
        robot_state: dict[str, object],
    ) -> NavigationObservation:
        self.sequence += 1
        self.calls.append((episode_id, reset, robot_state))
        frame = EncodedRGBFrame(
            sequence=self.sequence,
            captured_at_s=100.0 + self.sequence * 0.01,
            data=_JPEG,
            width=1,
            height=1,
        )
        return NavigationObservation(
            episode_id=episode_id,
            sequence=self.sequence,
            reset=reset,
            rgb_frames=(frame,),
            robot_state=robot_state,
        )


class _Control:
    def __init__(self, *, preflight_error: Exception | None = None) -> None:
        self.preflight_error = preflight_error
        self.preflight_calls = 0
        self.state_calls = 0
        self.started: list[BaseVelocityCommand] = []
        self.updated: list[BaseVelocityCommand] = []
        self.stop_calls = 0

    def _state(self) -> Go2State:
        return Go2State(
            pose=Pose2D(0.0, 0.0, 0.0),
            forward_velocity_mps=0.0,
            lateral_velocity_mps=0.0,
            yaw_rate_rps=0.0,
            sequence=self.state_calls,
            received_at_s=100.0 + self.state_calls * 0.001,
        )

    def preflight(self) -> Go2State:
        self.preflight_calls += 1
        if self.preflight_error is not None:
            raise self.preflight_error
        return self._state()

    def state(self, *, require_fresh: bool = True) -> Go2State:
        assert require_fresh
        self.state_calls += 1
        return self._state()

    def start_velocity_lease(
        self, command: BaseVelocityCommand, *, duration_s: float = 10.0
    ) -> str:
        assert duration_s == 10.0
        self.started.append(command)
        return "only-lease"

    def update_velocity_lease(self, action_id: str, command: BaseVelocityCommand) -> None:
        assert action_id == "only-lease"
        self.updated.append(command)

    def stop(self) -> None:
        self.stop_calls += 1

    @property
    def motion_writes(self) -> int:
        return len(self.started) + len(self.updated) + self.stop_calls


@dataclass
class _Provider:
    plans: list[WaypointPlan]
    delays_s: tuple[float, ...] = ()
    failure: Exception | None = None

    def __post_init__(self) -> None:
        self.requests: list[InferenceRequest] = []

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index < len(self.delays_s):
            await asyncio.sleep(self.delays_s[index])
        if self.failure is not None:
            raise self.failure
        plan = self.plans[index]
        return InferenceResult(request.request_id, plan)


def _session(
    provider: _Provider,
    camera: _Camera,
    control: _Control,
    *,
    execute: bool,
    max_runtime_s: float = 0.5,
) -> Go2NavigationSession:
    return Go2NavigationSession(
        provider,  # type: ignore[arg-type]
        camera,  # type: ignore[arg-type]
        control,  # type: ignore[arg-type]
        config=Go2NavigationSessionConfig(
            control_hz=100.0,
            max_runtime_s=max_runtime_s,
            execute=execute,
        ),
    )


def test_slow_inference_does_not_interrupt_fast_control_loop() -> None:
    camera = _Camera()
    control = _Control()
    provider = _Provider(
        [
            WaypointPlan(1, (Waypoint(1.0, 0.0, 0.0),)),
            WaypointPlan(2, (), terminal=True),
        ],
        delays_s=(0.01, 0.12),
    )

    result = asyncio.run(_session(provider, camera, control, execute=True).run("find tripod"))

    assert result.reason == "terminal"
    assert result.inference_count == 2
    assert result.control_ticks >= 8
    assert len(control.started) == 1
    assert len(control.updated) >= 5
    assert control.stop_calls == 1


def test_episode_is_fixed_reset_only_first_and_camera_sequence_increases() -> None:
    camera = _Camera()
    control = _Control()
    provider = _Provider(
        [
            WaypointPlan(1, (Waypoint(0.5, 0.0, 0.0),)),
            WaypointPlan(2, (), terminal=True),
        ]
    )

    result = asyncio.run(
        _session(provider, camera, control, execute=False).run(
            "find yellow wall", episode_id="go2:test-episode"
        )
    )

    observations = [request.payload.observation for request in provider.requests]
    assert all(isinstance(request.payload, NavigationRequest) for request in provider.requests)
    assert [item.episode_id for item in observations] == [
        "go2:test-episode",
        "go2:test-episode",
    ]
    assert [item.reset for item in observations] == [True, False]
    assert [item.sequence for item in observations] == [1, 2]
    assert result.last_observation_sequence == 2


def test_terminal_stop_ends_dry_run_without_any_motion_write() -> None:
    camera = _Camera()
    control = _Control()
    provider = _Provider([WaypointPlan(1, (), terminal=True)])

    result = asyncio.run(_session(provider, camera, control, execute=False).run("stop here"))

    assert result.reason == "terminal"
    assert result.plans_accepted == 1
    assert control.preflight_calls == 0
    assert control.motion_writes == 0
    assert any(event.kind == "terminal_reached" for event in result.events)


def test_execute_error_forces_stop_and_strictly_rejects_non_waypoint_output() -> None:
    camera = _Camera()
    control = _Control()
    provider = _Provider([])

    async def wrong_output(request: InferenceRequest) -> InferenceResult:
        return InferenceResult(request.request_id, {"action": "forward"})

    provider.infer_async = wrong_output  # type: ignore[method-assign]
    session = _session(provider, camera, control, execute=True)

    with pytest.raises(Go2NavigationSessionError, match="WaypointPlan"):
        asyncio.run(session.run("go forward"))

    assert control.preflight_calls == 1
    assert control.stop_calls == 1
    assert any(event.kind == "session_failed" for event in session.events)


def test_failed_preflight_has_zero_motion_writes() -> None:
    camera = _Camera()
    control = _Control(preflight_error=RuntimeError("operator not ready"))
    provider = _Provider([WaypointPlan(1, (), terminal=True)])

    with pytest.raises(RuntimeError, match="not ready"):
        asyncio.run(_session(provider, camera, control, execute=True).run("go"))

    assert not provider.requests
    assert not camera.calls
    assert control.motion_writes == 0


def test_max_runtime_stops_execute_session() -> None:
    camera = _Camera()
    control = _Control()
    provider = _Provider(
        [WaypointPlan(1, (Waypoint(1.0, 0.0, 0.0),))],
        delays_s=(0.15,),
    )

    async def run_and_observe_stop():
        task = asyncio.create_task(
            _session(provider, camera, control, execute=True, max_runtime_s=0.02).run("go")
        )
        await asyncio.sleep(0.06)
        assert control.stop_calls == 1
        assert not task.done()
        return await task

    result = asyncio.run(run_and_observe_stop())

    assert result.reason == "max_runtime"
    assert control.stop_calls == 1
