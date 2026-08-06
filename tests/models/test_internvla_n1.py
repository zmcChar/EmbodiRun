from __future__ import annotations

import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from embodied_runtime.models.vla.internvla_n1 import (
    VARIANT_DUALVLN,
    VARIANT_NAVDP,
    InternVLAConfig,
    InternVLADepthError,
    InternVLADepthPayload,
    InternVLAOutputError,
    InternVLARuntime,
    InternVLARuntimeError,
    NativePrediction,
    convert_native_prediction,
    decode_depth_payload,
)

RGB = np.array(
    [
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]],
    ],
    dtype=np.uint8,
)


def _depth_png(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


class FakeOfficialAgent:
    def __init__(self, args: object, results: list[object]) -> None:
        self.args = args
        self.results = results
        self.reset_calls = 0
        self.steps: list[dict[str, object]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        instruction: str,
        *,
        intrinsic: np.ndarray,
        look_down: bool,
    ) -> object:
        self.steps.append(
            {
                "rgb": rgb.copy(),
                "depth": depth.copy(),
                "pose": pose.copy(),
                "instruction": instruction,
                "intrinsic": intrinsic.copy(),
                "look_down": look_down,
            }
        )
        return self.results.pop(0)


def _official_result(
    *,
    action: object | None = None,
    trajectory: object | None = None,
) -> object:
    return SimpleNamespace(
        output_action=action,
        output_trajectory=trajectory,
        output_pixel=None,
    )


def test_config_preserves_variant_and_plan_step_gap_without_loading() -> None:
    config = InternVLAConfig(variant=VARIANT_NAVDP, plan_step_gap=7)
    runtime = InternVLARuntime(config, agent_factory=lambda args: None)

    assert runtime.variant == VARIANT_NAVDP
    assert runtime.spec.depth_required
    assert runtime.config.plan_step_gap == 7
    assert not runtime.loaded
    runtime.reset()
    assert not runtime.loaded


def test_dualvln_is_lazy_rgb_only_and_uses_cpu_fake_agent() -> None:
    holder: dict[str, FakeOfficialAgent] = {}

    def factory(args: object) -> FakeOfficialAgent:
        holder["agent"] = FakeOfficialAgent(args, [_official_result(action=[1])])
        return holder["agent"]

    runtime = InternVLARuntime(
        variant=VARIANT_DUALVLN,
        plan_step_gap=6,
        agent_factory=factory,
    )
    assert not runtime.loaded

    # Non-zero optional depth must not turn the RGB-only checkpoint into RGB-D.
    native = runtime.predict(RGB, np.full((2, 2), 9.0, dtype=np.float32), "go")

    assert runtime.loaded
    assert native.discrete_action == [1]
    agent = holder["agent"]
    assert agent.args.plan_step_gap == 6
    assert agent.args.model_path == "InternRobotics/InternVLA-N1-DualVLN"
    assert agent.reset_calls == 1
    assert agent.steps[0]["instruction"] == "go"
    depth = agent.steps[0]["depth"]
    assert isinstance(depth, np.ndarray)
    assert depth.dtype == np.float32
    assert np.all(depth == 0.0)


def test_navdp_requires_registered_depth_metres_and_passes_float32() -> None:
    holder: dict[str, FakeOfficialAgent] = {}

    def factory(args: object) -> FakeOfficialAgent:
        holder["agent"] = FakeOfficialAgent(args, [_official_result(action=[2])])
        return holder["agent"]

    runtime = InternVLARuntime(variant=VARIANT_NAVDP, agent_factory=factory)
    with pytest.raises(InternVLARuntimeError, match="requires registered depth"):
        runtime.predict(RGB, None, "turn left")

    native = runtime.predict(
        RGB,
        np.array([[0.0, 1.0], [2.5, 65.535]], dtype=np.float64),
        "turn left",
    )
    assert native.discrete_action == [2]
    depth = holder["agent"].steps[0]["depth"]
    assert isinstance(depth, np.ndarray)
    assert depth.dtype == np.float32
    assert depth.flags.c_contiguous
    assert depth[1, 0] == pytest.approx(2.5)


def test_runtime_consumes_exactly_one_look_down_retry() -> None:
    holder: dict[str, FakeOfficialAgent] = {}

    def factory(args: object) -> FakeOfficialAgent:
        holder["agent"] = FakeOfficialAgent(
            args,
            [_official_result(action=[5]), _official_result(action=[3])],
        )
        return holder["agent"]

    runtime = InternVLARuntime(agent_factory=factory)
    native = runtime.predict(RGB, None, "inspect then turn")

    assert native.discrete_action == [3]
    assert [step["look_down"] for step in holder["agent"].steps] == [False, True]


def test_runtime_rejects_repeated_look_down_instead_of_looping() -> None:
    def factory(args: object) -> FakeOfficialAgent:
        return FakeOfficialAgent(
            args,
            [_official_result(action=[5]), _official_result(action=[5])],
        )

    runtime = InternVLARuntime(agent_factory=factory)
    with pytest.raises(InternVLARuntimeError, match="repeatedly"):
        runtime.predict(RGB, None, "go")


def test_uint16_depth_png_uses_explicit_metre_scale() -> None:
    raw = np.array([[0, 1000], [2500, 65535]], dtype=np.uint16)
    payload = InternVLADepthPayload(
        data=_depth_png(raw),
        width=2,
        height=2,
        scale_m=0.001,
    )

    decoded = decode_depth_payload(payload, expected_shape=(2, 2))

    assert decoded.dtype == np.float32
    assert decoded[0, 0] == 0.0
    assert decoded[0, 1] == pytest.approx(1.0)
    assert decoded[1, 0] == pytest.approx(2.5)
    assert decoded[1, 1] == pytest.approx(65.535, abs=0.001)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"encoding": "8UC1"}, "uint16"),
        ({"registered_to_rgb": False}, "registered"),
        ({"scale_m": 0.0}, "scale_m"),
        ({"scale_m": "0.001"}, "number"),
    ],
)
def test_depth_rejects_ambiguous_encoding_registration_and_scale(
    updates: dict[str, object],
    message: str,
) -> None:
    raw = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    payload: dict[str, object] = {
        "data": _depth_png(raw),
        "width": 2,
        "height": 2,
        "scale_m": 0.001,
        "registered_to_rgb": True,
        "encoding": "uint16",
    }
    payload.update(updates)
    with pytest.raises(InternVLADepthError, match=message):
        decode_depth_payload(payload, expected_shape=(2, 2))


def test_depth_rejects_rgb_shape_mismatch() -> None:
    raw = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    payload = InternVLADepthPayload(
        data=_depth_png(raw),
        width=2,
        height=2,
        scale_m=0.001,
    )
    with pytest.raises(InternVLADepthError, match="match RGB"):
        decode_depth_payload(payload, expected_shape=(2, 3))


def test_trajectory_drops_warmup_prefix_and_uses_path_tangents() -> None:
    output = convert_native_prediction(
        NativePrediction(
            trajectory=[
                [0.0, 0.0],
                [0.02, 0.0],
                [0.05, 0.01],
                [0.10, 0.02],
                [0.30, 0.10],
            ]
        ),
        7,
    )

    assert not output.terminal
    assert [(point.x_m, point.y_m) for point in output.waypoints] == [
        (0.1, 0.02),
        (0.3, 0.1),
    ]
    expected_yaw = math.atan2(0.08, 0.2)
    assert output.waypoints[0].yaw_rad == pytest.approx(expected_yaw)
    assert output.waypoints[1].yaw_rad == pytest.approx(expected_yaw)
    assert output.as_dict()["kind"] == "waypoint_plan"


def test_discrete_actions_convert_to_go2_spatial_waypoints() -> None:
    output = convert_native_prediction(
        NativePrediction(discrete_action=np.array([1, 2, 1, 3])),
        2,
    )

    assert len(output.waypoints) == 4
    assert output.waypoints[0].x_m == pytest.approx(0.25)
    assert output.waypoints[1].yaw_rad == pytest.approx(math.radians(15.0))
    assert output.waypoints[2].x_m == pytest.approx(0.25 + 0.25 * math.cos(math.radians(15.0)))
    assert output.waypoints[2].y_m > 0.0
    assert output.waypoints[-1].yaw_rad == pytest.approx(0.0)


def test_stop_is_terminal_and_malformed_native_unions_are_rejected() -> None:
    stopped = convert_native_prediction(NativePrediction(discrete_action=[0]), 9)
    assert stopped.terminal
    assert stopped.waypoints == ()

    invalid = (
        NativePrediction(),
        NativePrediction(trajectory=[[0, 0]] * 4, discrete_action=[1]),
        NativePrediction(trajectory=[[0, 0]] * 3),
        NativePrediction(trajectory=[[0, 0]] * 3 + [[True, 0]]),
        NativePrediction(trajectory=[[0, 0]] * 3 + [["1", 0]]),
        NativePrediction(discrete_action=[0, 1]),
        NativePrediction(discrete_action=[5]),
        NativePrediction(discrete_action=[9]),
    )
    for native in invalid:
        with pytest.raises(InternVLAOutputError):
            convert_native_prediction(native, 0)
