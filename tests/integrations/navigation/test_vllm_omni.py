from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any

import numpy as np
import pytest

import embodied_runtime.integrations.navigation.vllm_omni.navila as navila_module
import embodied_runtime.integrations.navigation.vllm_omni.streamvln as streamvln_module
from embodied_runtime.engine import InferenceRequest, InferenceResult
from embodied_runtime.integrations.navigation.vllm_omni.codec import (
    NAVILA_ACTION_HEAD,
    STREAMVLN_ACTION_HEAD,
    decode_navila_action,
    decode_streamvln_actions,
    encode_navila_action,
    encode_streamvln_actions,
)
from embodied_runtime.integrations.navigation.vllm_omni.common import (
    NAVIGATION_PROTOCOL_NAME,
    NAVIGATION_PROTOCOL_VERSION,
    navigation_handshake_metadata,
    validate_image_payload,
    validate_navigation_handshake,
)
from embodied_runtime.integrations.navigation.vllm_omni.navila import (
    VllmOmniNaVILANavigationPolicy,
)
from embodied_runtime.integrations.navigation.vllm_omni.streamvln import (
    VllmOmniStreamVLNNavigationPolicy,
)
from embodied_runtime.models.vla.navila import NaVILAAction, NaVILAPrimitive
from embodied_runtime.tasks.navigation import (
    EncodedRGBFrame,
    NavigationObservation,
    NavigationRequest,
)


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _request(sequence: int, *, reset: bool = False, episode_id: str = "episode"):
    frame = EncodedRGBFrame(
        sequence=sequence,
        captured_at_s=float(sequence + 1),
        data=b"\xff\xd8placeholder",
        width=3,
        height=2,
    )
    return NavigationRequest(
        "find the doorway",
        NavigationObservation(
            episode_id=episode_id,
            sequence=sequence,
            reset=reset,
            rgb_frames=(frame,),
        ),
    )


class _Endpoint:
    def __init__(
        self,
        metadata: dict[str, Any],
        actions: dict[str, np.ndarray],
        *,
        session_id: str = "fake-client",
    ) -> None:
        self.metadata = metadata
        self.actions = actions
        self.session_id = session_id
        self.connect_count = 0
        self.requests: list[InferenceRequest] = []
        self.reset_calls: list[tuple[dict[str, Any], str | None]] = []
        self.closed = False
        self.fail_next = False

    async def connect(self):
        self.connect_count += 1
        return dict(self.metadata)

    async def reset(self, reset_info=None, *, session_id=None):
        self.reset_calls.append((dict(reset_info or {}), session_id))
        return "reset successful"

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected transport failure")
        return InferenceResult(
            request_id=request.request_id,
            output={"actions": self.actions},
            execution_time_s=0.25,
        )

    async def aclose(self) -> None:
        self.closed = True


def _metadata(model_family: str, horizon: int, dimension: int) -> dict[str, Any]:
    metadata = navigation_handshake_metadata(model_family, implementation="vllm_hybrid")
    metadata["action_horizon"] = horizon
    metadata["action_dim"] = dimension
    return metadata


def test_server_metadata_builder_rejects_unknown_models_and_implementations() -> None:
    metadata = navigation_handshake_metadata("streamvln", implementation="vllm_native")
    assert metadata["protocol_name"] == NAVIGATION_PROTOCOL_NAME
    assert metadata["protocol_version"] == NAVIGATION_PROTOCOL_VERSION
    assert metadata["padding_id"] == -1

    with pytest.raises(ValueError, match="unsupported.*model family"):
        navigation_handshake_metadata("internvla", implementation="vllm_native")
    with pytest.raises(ValueError, match="implementation"):
        navigation_handshake_metadata("navila", implementation="unknown")


def test_streamvln_wire_round_trip_preserves_short_sequences() -> None:
    encoded = encode_streamvln_actions((1, 2))

    assert encoded[STREAMVLN_ACTION_HEAD].dtype == np.float32
    assert encoded[STREAMVLN_ACTION_HEAD].shape == (1, 4, 1)
    assert encoded[STREAMVLN_ACTION_HEAD][0, :, 0].tolist() == [1.0, 2.0, -1.0, -1.0]
    assert decode_streamvln_actions(encoded) == (1, 2)


@pytest.mark.parametrize("actions", [(1,), (1, 2), (1, 2, 3, 1), (1, 0)])
def test_streamvln_wire_round_trip_covers_full_and_terminal_sequences(actions) -> None:
    assert decode_streamvln_actions(encode_streamvln_actions(actions)) == actions


def test_streamvln_wire_rejects_non_suffix_padding() -> None:
    malformed = {STREAMVLN_ACTION_HEAD: np.asarray([[[1], [-1], [2], [-1]]], dtype=np.float32)}

    with pytest.raises(ValueError, match="contiguous suffix"):
        decode_streamvln_actions(malformed)


@pytest.mark.parametrize(
    "action",
    [
        NaVILAAction(NaVILAPrimitive.STOP),
        NaVILAAction(NaVILAPrimitive.MOVE_FORWARD, 50, "cm"),
        NaVILAAction(NaVILAPrimitive.TURN_LEFT, 30, "degree"),
        NaVILAAction(NaVILAPrimitive.TURN_RIGHT, 45, "degree"),
    ],
)
def test_navila_wire_round_trip(action: NaVILAAction) -> None:
    encoded = encode_navila_action(action)

    assert encoded[NAVILA_ACTION_HEAD].dtype == np.float32
    assert encoded[NAVILA_ACTION_HEAD].shape == (1, 1, 3)
    assert decode_navila_action(encoded) == action


def test_wire_decoder_rejects_non_finite_values() -> None:
    actions = {NAVILA_ACTION_HEAD: np.asarray([[[1, np.nan, 1]]], dtype=np.float32)}

    with pytest.raises(ValueError, match="finite"):
        decode_navila_action(actions)


def test_navila_wire_rejects_fractional_fields_and_wrong_units() -> None:
    fractional = {NAVILA_ACTION_HEAD: np.asarray([[[1, 25.5, 1]]], dtype=np.float32)}
    wrong_unit = {NAVILA_ACTION_HEAD: np.asarray([[[1, 25, 2]]], dtype=np.float32)}

    with pytest.raises(ValueError, match="integer-valued"):
        decode_navila_action(fractional)
    with pytest.raises(ValueError, match="incompatible unit"):
        decode_navila_action(wrong_unit)


def test_handshake_rejects_wrong_model_and_boolean_dimensions() -> None:
    with pytest.raises(RuntimeError, match="model_family='streamvln'"):
        validate_navigation_handshake(
            _metadata("navila", 4, 1),
            model_family="streamvln",
            input_schema="rgb_uint8_hwc",
            action_head=STREAMVLN_ACTION_HEAD,
            action_horizon=4,
            action_dim=1,
            padding_id=-1,
        )
    with pytest.raises(RuntimeError, match="action_horizon=4"):
        validate_navigation_handshake(
            _metadata("streamvln", True, 1),
            model_family="streamvln",
            input_schema="rgb_uint8_hwc",
            action_head=STREAMVLN_ACTION_HEAD,
            action_horizon=4,
            action_dim=1,
            padding_id=-1,
        )


def test_handshake_rejects_missing_private_schema_fields() -> None:
    metadata = _metadata("navila", 1, 3)
    del metadata["action_head"]

    with pytest.raises(RuntimeError, match="action_head='navigation'"):
        validate_navigation_handshake(
            metadata,
            model_family="navila",
            input_schema="rgb_frames_uint8_list_hwc",
            action_head=NAVILA_ACTION_HEAD,
            action_horizon=1,
            action_dim=3,
        )


def test_image_payload_accepts_mixed_shapes_and_enforces_limit(monkeypatch) -> None:
    images = (
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((4, 5, 3), dtype=np.uint8),
    )
    assert validate_image_payload(images) == 78

    monkeypatch.setattr(
        "embodied_runtime.integrations.navigation.vllm_omni.common.MAX_OPENPI_IMAGE_PAYLOAD_BYTES",
        77,
    )
    with pytest.raises(ValueError, match="60 MiB"):
        validate_image_payload(images)


@async_test
async def test_streamvln_policy_resets_session_and_maps_remote_actions(monkeypatch) -> None:
    endpoint = _Endpoint(_metadata("streamvln", 4, 1), encode_streamvln_actions((1, 2)))
    monkeypatch.setattr(
        streamvln_module,
        "decode_rgb",
        lambda frame: np.full((2, 3, 3), frame.sequence, dtype=np.uint8),
    )
    policy = VllmOmniStreamVLNNavigationPolicy(endpoint=endpoint)

    await policy.prepare()
    plan = await policy.plan(_request(7, reset=True))

    assert endpoint.connect_count == 2
    assert endpoint.reset_calls == [({"episode_id": "episode"}, "fake-client")]
    assert endpoint.requests[0].payload["prompt"] == "find the doorway"
    assert endpoint.requests[0].payload["rgb"].shape == (2, 3, 3)
    assert endpoint.requests[0].payload["episode_id"] == "episode"
    assert endpoint.requests[0].payload["observation_sequence"] == 7
    assert endpoint.requests[0].payload["client_request_id"] == endpoint.requests[0].request_id
    assert endpoint.requests[0].metadata["session_id"] == "fake-client"
    assert endpoint.requests[0].deadline_s == pytest.approx(120.0)
    assert plan.observation_sequence == 7
    assert len(plan.waypoints) == 2
    assert not plan.terminal

    await policy.aclose()
    await policy.aclose()
    assert endpoint.closed


@async_test
async def test_navila_policy_samples_episode_history_for_remote_inference(monkeypatch) -> None:
    endpoint = _Endpoint(
        _metadata("navila", 1, 3),
        encode_navila_action(NaVILAAction(NaVILAPrimitive.MOVE_FORWARD, 25, "cm")),
    )
    monkeypatch.setattr(
        navila_module,
        "decode_rgb",
        lambda frame: np.full((2 + frame.sequence % 2, 3, 3), frame.sequence, dtype=np.uint8),
    )
    policy = VllmOmniNaVILANavigationPolicy(endpoint=endpoint)

    for sequence in range(10):
        plan = await policy.plan(_request(sequence, reset=sequence == 0))
        assert plan.waypoints[0].x_m == pytest.approx(0.25)

    assert endpoint.connect_count == 10
    assert len(endpoint.reset_calls) == 1
    assert endpoint.reset_calls[0][1] == "fake-client"
    assert policy.history_size == 10
    final_frames = endpoint.requests[-1].payload["rgb_frames"]
    assert isinstance(final_frames, tuple)
    assert [frame.shape[0] for frame in final_frames] == [2, 3, 2, 3, 3, 2, 3, 3]
    assert [int(frame[0, 0, 0]) for frame in final_frames] == [0, 1, 2, 3, 5, 6, 7, 9]

    await policy.aclose()
    assert policy.history_size == 0
    assert endpoint.closed


@async_test
async def test_policy_closes_endpoint_after_contract_mismatch() -> None:
    endpoint = _Endpoint(_metadata("navila", 1, 3), encode_streamvln_actions((0,)))
    policy = VllmOmniStreamVLNNavigationPolicy(endpoint=endpoint)

    with pytest.raises(RuntimeError, match="model_family='streamvln'"):
        await policy.prepare()

    assert endpoint.closed


@async_test
async def test_streamvln_failure_poisons_episode_until_explicit_reset(monkeypatch) -> None:
    endpoint = _Endpoint(_metadata("streamvln", 4, 1), encode_streamvln_actions((1,)))
    monkeypatch.setattr(
        streamvln_module,
        "decode_rgb",
        lambda frame: np.full((2, 3, 3), frame.sequence, dtype=np.uint8),
    )
    policy = VllmOmniStreamVLNNavigationPolicy(endpoint=endpoint)
    await policy.plan(_request(0, reset=True))

    endpoint.fail_next = True
    with pytest.raises(RuntimeError, match="injected transport failure"):
        await policy.plan(_request(1))
    request_count = len(endpoint.requests)
    with pytest.raises(RuntimeError, match="state is uncertain"):
        await policy.plan(_request(1))
    assert len(endpoint.requests) == request_count

    plan = await policy.plan(_request(1, reset=True))
    assert plan.observation_sequence == 1
    assert len(endpoint.reset_calls) == 2
    await policy.aclose()


def test_default_clients_use_isolated_sessions_and_disable_replay() -> None:
    first = VllmOmniStreamVLNNavigationPolicy()
    second = VllmOmniStreamVLNNavigationPolicy()

    assert first.endpoint.session_id != second.endpoint.session_id
    assert first.endpoint.reconnect_attempts == 0
    assert second.endpoint.reconnect_attempts == 0
