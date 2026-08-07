from __future__ import annotations

import asyncio

import pytest

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
