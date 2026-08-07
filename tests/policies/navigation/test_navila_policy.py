from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest
from PIL import Image

from embodied_runtime.models.vla.navila import (
    NaVILAAction,
    NaVILAPrediction,
    NaVILAPrimitive,
)
from embodied_runtime.policies.navigation.navila import (
    NaVILANavigationPolicy,
    navila_prediction_to_waypoint_plan,
)
from embodied_runtime.tasks.navigation import (
    EncodedRGBFrame,
    NavigationObservation,
    NavigationRequest,
)


def _prediction(
    primitive: NaVILAPrimitive,
    magnitude: int | None = None,
    unit: str | None = None,
) -> NaVILAPrediction:
    return NaVILAPrediction(
        NaVILAAction(primitive, magnitude, unit),  # type: ignore[arg-type]
        primitive.value,
        (1,),
        0.1,
    )


@pytest.mark.parametrize(
    ("prediction", "x_m", "yaw_rad", "terminal"),
    [
        (_prediction(NaVILAPrimitive.STOP), None, None, True),
        (_prediction(NaVILAPrimitive.MOVE_FORWARD, 75, "cm"), 0.75, 0.0, False),
        (_prediction(NaVILAPrimitive.TURN_LEFT, 30, "degree"), 0.0, np.pi / 6, False),
        (_prediction(NaVILAPrimitive.TURN_RIGHT, 45, "degree"), 0.0, -np.pi / 4, False),
    ],
)
def test_native_action_maps_to_one_midlevel_waypoint(prediction, x_m, yaw_rad, terminal) -> None:
    plan = navila_prediction_to_waypoint_plan(prediction, 9)
    assert plan.observation_sequence == 9
    assert plan.terminal is terminal
    if terminal:
        assert plan.waypoints == ()
    else:
        assert len(plan.waypoints) == 1
        assert plan.waypoints[0].x_m == pytest.approx(x_m)
        assert plan.waypoints[0].yaw_rad == pytest.approx(yaw_rad)


def _jpeg(value: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), (value, value, value)).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _request(sequence: int, *, reset: bool = False, episode: str = "episode"):
    return NavigationRequest(
        "find the target",
        NavigationObservation(
            episode_id=episode,
            sequence=sequence,
            reset=reset,
            rgb_frames=(
                EncodedRGBFrame(
                    sequence,
                    float(sequence + 1),
                    _jpeg(sequence * 20),
                    width=4,
                    height=4,
                ),
            ),
        ),
    )


class _Runtime:
    def __init__(self) -> None:
        self.loads = 0
        self.closed = 0
        self.calls: list[tuple[tuple[int, ...], str]] = []
        self.fail_next = False

    def load(self):
        self.loads += 1
        return self

    def predict(self, frames, instruction):
        values = tuple(int(frame[0, 0, 0]) for frame in frames)
        self.calls.append((values, instruction))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected failure")
        return _prediction(NaVILAPrimitive.MOVE_FORWARD, 25, "cm")

    def close(self):
        self.closed += 1


def test_policy_prepares_samples_full_episode_and_closes() -> None:
    async def run() -> None:
        runtime = _Runtime()
        policy = NaVILANavigationPolicy(runtime=runtime)  # type: ignore[arg-type]
        await policy.prepare()
        for sequence in range(10):
            plan = await policy.plan(_request(sequence, reset=sequence == 0))
            assert plan.waypoints[0].x_m == pytest.approx(0.25)
        assert runtime.loads == 1
        assert policy.history_size == 10
        assert runtime.calls[-1][0] == (0, 20, 40, 60, 100, 120, 140, 180)
        await policy.aclose()
        await policy.aclose()
        assert runtime.closed == 1

    asyncio.run(run())


def test_failed_inference_does_not_commit_episode_history() -> None:
    async def run() -> None:
        runtime = _Runtime()
        runtime.fail_next = True
        policy = NaVILANavigationPolicy(runtime=runtime)  # type: ignore[arg-type]
        request = _request(0, reset=True)
        with pytest.raises(RuntimeError, match="injected"):
            await policy.plan(request)
        assert policy.history_size == 0
        await policy.plan(request)
        assert policy.history_size == 1
        assert [len(values) for values, _ in runtime.calls] == [1, 1]
        await policy.aclose()

    asyncio.run(run())
