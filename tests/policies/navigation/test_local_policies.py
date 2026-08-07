from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from embodied_runtime.models.vla.internvla_n1 import NativePrediction
from embodied_runtime.policies.navigation.errors import NavigationPolicyError
from embodied_runtime.policies.navigation.internvla import InternVLANavigationPolicy
from embodied_runtime.policies.navigation.streamvln import StreamVLNNavigationPolicy
from embodied_runtime.tasks.navigation import (
    EncodedRGBFrame,
    NavigationObservation,
    NavigationRequest,
    WaypointPlan,
)


def _jpeg() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 200, 0)).save(stream, format="JPEG")
    return stream.getvalue()


def _request(
    sequence: int,
    *,
    episode: str = "episode-1",
    reset: bool = False,
) -> NavigationRequest:
    observation = NavigationObservation(
        episode_id=episode,
        sequence=sequence,
        reset=reset,
        rgb_frames=(
            EncodedRGBFrame(
                sequence=sequence,
                captured_at_s=float(sequence + 1),
                data=_jpeg(),
                width=2,
                height=2,
            ),
        ),
    )
    return NavigationRequest("go to the yellow post", observation)


class _StreamRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_path="stream-fixture",
            revision="test",
            num_future_steps=4,
        )
        self.loads = 0
        self.resets = 0
        self.inputs: list[np.ndarray] = []

    def load(self) -> _StreamRuntime:
        self.loads += 1
        return self

    def reset(self) -> None:
        self.resets += 1

    def predict(self, rgb: np.ndarray, instruction: str):
        self.inputs.append(rgb)
        return SimpleNamespace(actions=(1,))


class _InternRuntime:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            model_id="intern-fixture",
            name="dualvln",
            system1="fake",
            depth_required=False,
        )
        self.variant = "dualvln"
        self.loads = 0
        self.resets = 0

    def load(self) -> _InternRuntime:
        self.loads += 1
        return self

    def reset(self) -> None:
        self.resets += 1

    def predict(self, rgb: np.ndarray, depth: object, instruction: str) -> NativePrediction:
        assert rgb.shape == (2, 2, 3)
        assert depth is None
        return NativePrediction(discrete_action=[1])


def test_streamvln_policy_maps_prediction_and_preserves_episode_order() -> None:
    runtime = _StreamRuntime()
    policy = StreamVLNNavigationPolicy(runtime=runtime)  # type: ignore[arg-type]

    asyncio.run(policy.prepare())
    assert runtime.loads == 1
    plan = asyncio.run(policy.plan(_request(1, reset=True)))
    assert isinstance(plan, WaypointPlan)
    assert plan.waypoints[0].x_m == pytest.approx(0.25)
    assert runtime.resets == 1

    with pytest.raises(NavigationPolicyError, match="increase"):
        asyncio.run(policy.plan(_request(1)))


def test_internvla_policy_maps_native_output_and_requires_reset_on_new_episode() -> None:
    runtime = _InternRuntime()
    policy = InternVLANavigationPolicy(runtime=runtime)  # type: ignore[arg-type]

    asyncio.run(policy.prepare())
    assert runtime.loads == 1
    plan = asyncio.run(policy.plan(_request(1)))
    assert isinstance(plan, WaypointPlan)
    assert not plan.terminal
    assert len(plan.waypoints) == 1
    assert runtime.resets == 1

    with pytest.raises(NavigationPolicyError, match="reset=true"):
        asyncio.run(policy.plan(_request(2, episode="episode-2")))
