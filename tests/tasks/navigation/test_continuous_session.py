from __future__ import annotations

import asyncio

import pytest

from embodied_runtime.tasks.navigation import (
    NavigationRequest,
    NavigationSession,
    NavigationSessionConfig,
    NavigationSessionError,
    Waypoint,
    WaypointPlan,
)

from .session_fakes import MobileBase, ObservationSource, Policy


def test_continuous_session_uses_task_policy_without_inference_envelope() -> None:
    source = ObservationSource()
    base = MobileBase()
    policy = Policy(
        [
            WaypointPlan(1, (Waypoint(1, 0, 0),)),
            WaypointPlan(2, (), terminal=True),
        ],
        delays=(0.01, 0.08),
    )
    session = NavigationSession(
        policy,
        source,
        base,
        config=NavigationSessionConfig(control_hz=100, max_runtime_s=0.5, execute=True),
    )

    result = asyncio.run(session.run("find tripod", episode_id="task-nav"))

    assert result.reason == "terminal"
    assert all(isinstance(request, NavigationRequest) for request in policy.requests)
    assert source.resets == [True, False]
    assert base.started and base.updated
    assert base.stop_calls == 1


def test_continuous_dry_run_has_no_motion_or_preflight() -> None:
    base = MobileBase()
    session = NavigationSession(
        Policy([WaypointPlan(1, (), terminal=True)]),
        ObservationSource(),
        base,
        config=NavigationSessionConfig(control_hz=100, max_runtime_s=0.5),
    )
    result = asyncio.run(session.run("stop"))

    assert result.reason == "terminal"
    assert base.preflight_calls == base.stop_calls == 0
    assert base.started == []


def test_continuous_max_runtime_forces_one_final_stop() -> None:
    class SlowPolicy:
        async def plan(self, request):
            await asyncio.sleep(0.15)
            return WaypointPlan(request.observation.sequence, (Waypoint(1, 0, 0),))

    base = MobileBase()
    session = NavigationSession(
        SlowPolicy(),  # type: ignore[arg-type]
        ObservationSource(),
        base,
        config=NavigationSessionConfig(execute=True, max_runtime_s=0.02, control_hz=100),
    )

    async def run_and_observe_stop():
        task = asyncio.create_task(session.run("find target"))
        await asyncio.sleep(0.06)
        assert base.stop_calls == 1
        assert not task.done()
        return await task

    result = asyncio.run(run_and_observe_stop())

    assert result.reason == "max_runtime"
    assert base.preflight_calls == 1
    assert base.stop_calls == 1


def test_continuous_wrong_plan_type_forces_final_stop() -> None:
    class BadPolicy:
        async def plan(self, request):
            return {"not": "a plan"}

    base = MobileBase()
    session = NavigationSession(
        BadPolicy(),  # type: ignore[arg-type]
        ObservationSource(),
        base,
        config=NavigationSessionConfig(execute=True, max_runtime_s=1, control_hz=100),
    )
    with pytest.raises(NavigationSessionError, match="WaypointPlan"):
        asyncio.run(session.run("go"))
    assert base.stop_calls == 1
