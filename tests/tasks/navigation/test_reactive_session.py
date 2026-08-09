from __future__ import annotations

import asyncio
import math

import pytest

import embodied_runtime.tasks.navigation.reactive_session as reactive_module
from embodied_runtime.tasks.navigation import (
    MobileBaseError,
    NavigationSessionError,
    ReactiveNavigationSession,
    ReactiveNavigationSessionConfig,
    Waypoint,
    WaypointPlan,
)

from .session_fakes import MobileBase, ObservationSource, Policy


def test_reactive_terminal_waypoint_executes_then_recaptures() -> None:
    source = ObservationSource()
    base = MobileBase()
    policy = Policy(
        [
            WaypointPlan(1, (Waypoint(0.005, 0, 0),), terminal=True),
            WaypointPlan(2, (), terminal=True),
        ]
    )
    session = ReactiveNavigationSession(
        policy,
        source,
        base,
        config=ReactiveNavigationSessionConfig(
            execute=True,
            max_runtime_s=1,
            min_pulse_s=0.01,
            max_pulse_s=0.05,
            settle_s=0,
        ),
    )

    result = asyncio.run(session.run("find target"))

    assert result.reason == "terminal"
    assert result.inference_count == 2
    assert len(base.started) == 1
    assert source.resets == [True, False]
    assert base.stop_calls == 1


def test_reactive_executes_cumulative_waypoints_as_se2_relative_pulses(monkeypatch) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(reactive_module.asyncio, "sleep", no_sleep)
    source = ObservationSource()
    base = MobileBase()
    quarter_turn = math.pi / 2
    policy = Policy(
        [
            WaypointPlan(
                1,
                (
                    Waypoint(0, 0, quarter_turn),
                    Waypoint(0, 0.25, quarter_turn),
                    Waypoint(0, 0.25, 0),
                ),
                terminal=True,
            )
        ]
    )
    session = ReactiveNavigationSession(
        policy,
        source,
        base,
        config=ReactiveNavigationSessionConfig(
            execute=True,
            max_runtime_s=1,
            min_pulse_s=0.01,
            max_pulse_s=3,
            settle_s=0,
            max_waypoints_per_observation=3,
            terminal_after_waypoints=True,
        ),
    )

    result = asyncio.run(session.run("turn, move forward, then face ahead"))

    assert result.reason == "terminal"
    assert result.inference_count == 1
    assert result.motion_commands == 3
    assert source.resets == [True]
    assert len(base.started) == 3
    first, second, third = (item[0] for item in base.started)
    assert first.vx_mps == pytest.approx(0)
    assert first.vy_mps == pytest.approx(0)
    assert first.yaw_rate_rps > 0
    # The second cumulative target is +Y in the capture frame, but after the
    # first 90-degree turn it is straight ahead in the segment's local frame.
    assert second.vx_mps > 0
    assert second.vy_mps == pytest.approx(0, abs=1e-12)
    assert second.yaw_rate_rps == pytest.approx(0)
    assert third.vx_mps == pytest.approx(0)
    assert third.vy_mps == pytest.approx(0)
    assert third.yaw_rate_rps < 0
    assert [event.waypoint_index for event in result.events if event.kind == "pulse_started"] == [
        0,
        1,
        2,
    ]
    assert base.stop_calls == 1


def test_terminal_chunk_only_finishes_when_all_selected_waypoints_execute(monkeypatch) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(reactive_module.asyncio, "sleep", no_sleep)
    source = ObservationSource()
    base = MobileBase()
    policy = Policy(
        [
            WaypointPlan(
                1,
                (Waypoint(0.005, 0, 0), Waypoint(0.01, 0, 0), Waypoint(0.015, 0, 0)),
                terminal=True,
            ),
            WaypointPlan(2, (), terminal=True),
        ]
    )
    session = ReactiveNavigationSession(
        policy,
        source,
        base,
        config=ReactiveNavigationSessionConfig(
            execute=True,
            max_runtime_s=1,
            min_pulse_s=0.01,
            max_pulse_s=0.05,
            settle_s=0,
            max_waypoints_per_observation=2,
            terminal_after_waypoints=True,
        ),
    )

    result = asyncio.run(session.run("move forward"))

    assert result.reason == "terminal"
    assert result.inference_count == 2
    assert len(base.started) == 2
    assert source.resets == [True, False]


def test_reactive_retries_typed_mobile_base_error() -> None:
    class FlakyBase(MobileBase):
        def state(self):
            self.state_calls += 1
            if self.state_calls == 1:
                raise MobileBaseError("temporary")
            return self._state()

    base = FlakyBase()
    session = ReactiveNavigationSession(
        Policy([WaypointPlan(1, (), terminal=True)]),
        ObservationSource(),
        base,
        config=ReactiveNavigationSessionConfig(
            max_runtime_s=1,
            state_attempts=2,
            state_retry_delay_s=0,
        ),
    )

    result = asyncio.run(session.run("find target"))
    assert result.reason == "terminal"
    assert base.state_calls == 2
    assert any(event.kind == "state_retry" for event in result.events)


def test_reactive_persistent_state_error_stops_after_configured_attempts() -> None:
    class FailingBase(MobileBase):
        def state(self):
            self.state_calls += 1
            raise MobileBaseError("persistent state timeout")

    base = FailingBase()
    session = ReactiveNavigationSession(
        Policy([WaypointPlan(1, (), terminal=True)]),
        ObservationSource(),
        base,
        config=ReactiveNavigationSessionConfig(
            max_runtime_s=1,
            state_attempts=3,
            state_retry_delay_s=0,
        ),
    )

    with pytest.raises(MobileBaseError, match="persistent state timeout"):
        asyncio.run(session.run("find target"))

    assert base.state_calls == 3
    assert sum(event.kind == "state_retry" for event in session.events) == 2
    assert base.stop_calls == 0


def test_reactive_state_retry_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="state_attempts"):
        ReactiveNavigationSessionConfig(state_attempts=0)
    with pytest.raises(ValueError, match="state_retry_delay_s"):
        ReactiveNavigationSessionConfig(state_retry_delay_s=-0.1)
    with pytest.raises(ValueError, match="max_waypoints_per_observation"):
        ReactiveNavigationSessionConfig(max_waypoints_per_observation=0)
    with pytest.raises(TypeError, match="terminal_after_waypoints"):
        ReactiveNavigationSessionConfig(terminal_after_waypoints=1)  # type: ignore[arg-type]

    defaults = ReactiveNavigationSessionConfig()
    assert defaults.max_waypoints_per_observation == 1
    assert defaults.terminal_after_waypoints is False


def test_reactive_dry_run_never_writes_motion() -> None:
    base = MobileBase()
    session = ReactiveNavigationSession(
        Policy([WaypointPlan(1, (), terminal=True)]),
        ObservationSource(),
        base,
        config=ReactiveNavigationSessionConfig(execute=False, max_runtime_s=1),
    )

    result = asyncio.run(session.run("stop"))

    assert result.reason == "terminal"
    assert base.preflight_calls == 0
    assert base.started == []
    assert base.updated == []
    assert base.stop_calls == 0


def test_bad_policy_output_forces_final_stop() -> None:
    class BadPolicy:
        async def plan(self, request):
            return {"action": "forward"}

    base = MobileBase()
    session = ReactiveNavigationSession(
        BadPolicy(),  # type: ignore[arg-type]
        ObservationSource(),
        base,
        config=ReactiveNavigationSessionConfig(execute=True, max_runtime_s=1),
    )
    with pytest.raises(NavigationSessionError, match="WaypointPlan"):
        asyncio.run(session.run("go"))
    assert base.stop_calls == 1


def test_reactive_timeout_forces_final_stop() -> None:
    class SlowPolicy:
        async def plan(self, request):
            await asyncio.sleep(10)

    base = MobileBase()
    session = ReactiveNavigationSession(
        SlowPolicy(),  # type: ignore[arg-type]
        ObservationSource(),
        base,
        config=ReactiveNavigationSessionConfig(execute=True, max_runtime_s=0.02),
    )

    result = asyncio.run(session.run("find target"))

    assert result.reason == "max_runtime"
    assert base.stop_calls == 1


def test_failed_preflight_never_stops_or_touches_policy() -> None:
    base = MobileBase(preflight_error=RuntimeError("not ready"))
    policy = Policy([WaypointPlan(1, (), terminal=True)])
    session = ReactiveNavigationSession(
        policy,
        ObservationSource(),
        base,
        config=ReactiveNavigationSessionConfig(execute=True),
    )
    with pytest.raises(RuntimeError, match="not ready"):
        asyncio.run(session.run("go"))
    assert not policy.requests
    assert base.stop_calls == 0
