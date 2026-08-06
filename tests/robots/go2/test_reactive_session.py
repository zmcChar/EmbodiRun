from __future__ import annotations

import asyncio
import math

import pytest

from embodied_runtime.contracts import (
    EncodedRGBFrame,
    InferenceResult,
    NavigationObservation,
    Waypoint,
    WaypointPlan,
)
from embodied_runtime.robots.go2.client import Go2ClientError
from embodied_runtime.robots.go2.reactive_session import (
    Go2ReactiveNavigationSession,
    Go2ReactiveSessionConfig,
    waypoint_to_velocity_pulse,
)
from embodied_runtime.robots.go2.types import BaseVelocityCommand, Go2State
from embodied_runtime.utils import Pose2D

_JPEG = b"\xff\xd8\xff\xd9"


class _Control:
    def __init__(self) -> None:
        self.preflight_calls = 0
        self.state_calls = 0
        self.started: list[tuple[BaseVelocityCommand, float]] = []
        self.stop_calls = 0

    def _state(self) -> Go2State:
        return Go2State(Pose2D(0, 0, 0), 0, 0, 0, self.state_calls, 100.0)

    def preflight(self) -> Go2State:
        self.preflight_calls += 1
        return self._state()

    def state(self) -> Go2State:
        self.state_calls += 1
        return self._state()

    def start_velocity_lease(self, command: BaseVelocityCommand, *, duration_s: float) -> str:
        self.started.append((command, duration_s))
        return f"pulse-{len(self.started)}"

    def stop(self) -> None:
        self.stop_calls += 1


class _Camera:
    def __init__(self) -> None:
        self.sequence = 0
        self.resets: list[bool] = []

    def capture(self, *, episode_id, reset, robot_state):
        self.sequence += 1
        self.resets.append(reset)
        frame = EncodedRGBFrame(
            self.sequence,
            100.0 + self.sequence,
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


class _Provider:
    def __init__(self, plans: list[WaypointPlan]) -> None:
        self.plans = plans
        self.calls = 0

    async def infer_async(self, request):
        plan = self.plans[self.calls]
        self.calls += 1
        return InferenceResult(request.request_id, plan)


def _config(*, execute: bool, max_runtime_s: float = 1.0):
    return Go2ReactiveSessionConfig(
        execute=execute,
        max_runtime_s=max_runtime_s,
        min_pulse_s=0.01,
        max_pulse_s=0.05,
        settle_s=0,
    )


def test_streamvln_forward_and_turn_convert_to_bounded_pulses() -> None:
    forward, forward_s = waypoint_to_velocity_pulse(Waypoint(0.25, 0, 0))
    turn, turn_s = waypoint_to_velocity_pulse(Waypoint(0, 0, -math.radians(15)))

    assert forward.vx_mps == pytest.approx(0.30)
    assert forward_s == pytest.approx(0.25 / 0.30)
    assert turn.yaw_rate_rps == pytest.approx(-0.60)
    assert turn_s == pytest.approx(math.radians(15) / 0.60)


def test_terminal_with_waypoint_executes_one_pulse_then_recaptures() -> None:
    control = _Control()
    camera = _Camera()
    provider = _Provider(
        [
            WaypointPlan(1, (Waypoint(0.005, 0, 0),), terminal=True),
            WaypointPlan(2, (), terminal=True),
        ]
    )
    session = Go2ReactiveNavigationSession(
        provider,
        camera,
        control,
        config=_config(execute=True),  # type: ignore[arg-type]
    )

    result = asyncio.run(session.run("find target", episode_id="reactive-test"))

    assert result.reason == "terminal"
    assert result.inference_count == 2
    assert result.motion_commands == 1
    assert camera.resets == [True, False]
    assert control.preflight_calls == 1
    assert len(control.started) == 1
    assert control.stop_calls == 1


def test_dry_run_never_writes_control() -> None:
    control = _Control()
    session = Go2ReactiveNavigationSession(
        _Provider([WaypointPlan(1, (), terminal=True)]),  # type: ignore[arg-type]
        _Camera(),  # type: ignore[arg-type]
        control,  # type: ignore[arg-type]
        config=_config(execute=False),
    )

    result = asyncio.run(session.run("stop"))

    assert result.reason == "terminal"
    assert control.preflight_calls == 0
    assert control.started == []
    assert control.stop_calls == 0


def test_transient_state_timeout_is_retried_before_capture() -> None:
    class FlakyControl(_Control):
        def state(self) -> Go2State:
            self.state_calls += 1
            if self.state_calls == 1:
                raise Go2ClientError("temporary state timeout")
            return self._state()

    control = FlakyControl()
    session = Go2ReactiveNavigationSession(
        _Provider([WaypointPlan(1, (), terminal=True)]),  # type: ignore[arg-type]
        _Camera(),  # type: ignore[arg-type]
        control,  # type: ignore[arg-type]
        config=Go2ReactiveSessionConfig(
            execute=False,
            max_runtime_s=1,
            state_attempts=2,
            state_retry_delay_s=0,
        ),
    )

    result = asyncio.run(session.run("find target"))

    assert result.reason == "terminal"
    assert control.state_calls == 2
    assert any(event.kind == "state_retry" for event in result.events)


def test_persistent_state_timeout_stops_after_configured_attempts() -> None:
    class FailingControl(_Control):
        def state(self) -> Go2State:
            self.state_calls += 1
            raise Go2ClientError("persistent state timeout")

    control = FailingControl()
    session = Go2ReactiveNavigationSession(
        _Provider([WaypointPlan(1, (), terminal=True)]),  # type: ignore[arg-type]
        _Camera(),  # type: ignore[arg-type]
        control,  # type: ignore[arg-type]
        config=Go2ReactiveSessionConfig(
            execute=False,
            max_runtime_s=1,
            state_attempts=3,
            state_retry_delay_s=0,
        ),
    )

    with pytest.raises(Go2ClientError, match="persistent state timeout"):
        asyncio.run(session.run("find target"))

    assert control.state_calls == 3


def test_state_retry_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="state_attempts"):
        Go2ReactiveSessionConfig(state_attempts=0)
    with pytest.raises(ValueError, match="state_retry_delay_s"):
        Go2ReactiveSessionConfig(state_retry_delay_s=-0.1)


def test_timeout_forces_one_final_stop() -> None:
    class SlowProvider:
        async def infer_async(self, request):
            await asyncio.sleep(10)

    control = _Control()
    session = Go2ReactiveNavigationSession(
        SlowProvider(),  # type: ignore[arg-type]
        _Camera(),  # type: ignore[arg-type]
        control,  # type: ignore[arg-type]
        config=_config(execute=True, max_runtime_s=0.02),
    )

    result = asyncio.run(session.run("find target"))

    assert result.reason == "max_runtime"
    assert control.stop_calls == 1
