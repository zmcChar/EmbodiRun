from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from embodied_runtime.contracts import (
    EncodedRGBFrame,
    InferenceRequest,
    NavigationObservation,
    NavigationRequest,
    WaypointPlan,
)
from embodied_runtime.integrations.serving.navigation import (
    InternVLANavigationProvider,
    NavigationProviderError,
    StreamVLNNavigationProvider,
)
from embodied_runtime.models.vla.internvla_n1 import NativePrediction
from embodied_runtime.models.vln.streamvln import StreamVLNWaypoint, StreamVLNWaypointPlan


def _jpeg() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 200, 0)).save(stream, format="JPEG")
    return stream.getvalue()


def _request(sequence: int, *, episode: str = "episode-1", reset: bool = False):
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
    return InferenceRequest(NavigationRequest("go to the yellow post", observation))


class _StreamRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_path="stream-fixture",
            revision="test",
            num_future_steps=4,
        )
        self.resets = 0
        self.inputs: list[np.ndarray] = []

    def reset(self) -> None:
        self.resets += 1

    def predict(self, rgb: np.ndarray, instruction: str):
        self.inputs.append(rgb)
        return SimpleNamespace(
            waypoints=(StreamVLNWaypoint(0.25, 0.0, 0.0),),
            terminal=False,
            plan=StreamVLNWaypointPlan((StreamVLNWaypoint(0.25, 0.0, 0.0),), False),
        )


class _InternRuntime:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            model_id="intern-fixture",
            name="dualvln",
            system1="fake",
            depth_required=False,
        )
        self.variant = "dualvln"
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict(self, rgb: np.ndarray, depth: object, instruction: str) -> NativePrediction:
        assert rgb.shape == (2, 2, 3)
        assert depth is None
        return NativePrediction(discrete_action=[1])


def test_streamvln_provider_maps_prediction_and_preserves_episode_order() -> None:
    runtime = _StreamRuntime()
    provider = StreamVLNNavigationProvider(runtime=runtime)  # type: ignore[arg-type]

    result = asyncio.run(provider.infer_async(_request(1, reset=True)))
    assert isinstance(result.output, WaypointPlan)
    assert result.output.waypoints[0].x_m == pytest.approx(0.25)
    assert runtime.resets == 1

    with pytest.raises(NavigationProviderError, match="increase"):
        asyncio.run(provider.infer_async(_request(1)))


def test_internvla_provider_maps_native_output_and_requires_reset_on_new_episode() -> None:
    runtime = _InternRuntime()
    provider = InternVLANavigationProvider(runtime=runtime)  # type: ignore[arg-type]

    result = asyncio.run(provider.infer_async(_request(1)))
    assert isinstance(result.output, WaypointPlan)
    assert not result.output.terminal
    assert len(result.output.waypoints) == 1
    assert runtime.resets == 1

    with pytest.raises(NavigationProviderError, match="reset=true"):
        asyncio.run(provider.infer_async(_request(2, episode="episode-2")))
