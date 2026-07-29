from __future__ import annotations

import asyncio
import pickle
import time
from collections import deque
from functools import wraps
from typing import Any

import numpy as np
import pytest

from embodied_runtime.contracts import InferenceRequest, RawRequest
from embodied_runtime.distributed.communication import (
    AsyncRemoteInferenceEndpoint,
    OpenPIProtocolError,
    OpenPITimeoutError,
    OpenPiWebSocketEndpoint,
    validate_openpi_actions,
)
from embodied_runtime.distributed.communication.openpi import _websocket_connect_options


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class _PickleCodec:
    def pack(self, value: Any) -> bytes:
        return pickle.dumps(value)

    def unpack(self, value: bytes | str) -> Any:
        assert isinstance(value, bytes)
        return pickle.loads(value)


class _FakeConnection:
    def __init__(
        self,
        responses: list[Any],
        codec: _PickleCodec,
        *,
        receive_error_at: int | None = None,
        response_delay_s: float = 0.0,
    ) -> None:
        self._responses = deque(codec.pack(response) for response in responses)
        self._codec = codec
        self._receive_error_at = receive_error_at
        self._response_delay_s = response_delay_s
        self.receive_count = 0
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def send(self, message: bytes) -> None:
        self.sent.append(self._codec.unpack(message))

    def recv(self, timeout: float | None = None) -> bytes:
        self.receive_count += 1
        if self._receive_error_at == self.receive_count:
            raise ConnectionError("simulated broken websocket")
        if self._response_delay_s and self.receive_count > 1:
            time.sleep(self._response_delay_s)
        if not self._responses:
            raise AssertionError("fake connection has no queued response")
        return self._responses.popleft()

    def close(self) -> None:
        self.closed = True


class _ConnectionFactory:
    def __init__(self, connections: list[_FakeConnection]) -> None:
        self.connections = deque(connections)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, uri: str, timeout_s: float) -> _FakeConnection:
        self.calls.append((uri, timeout_s))
        if not self.connections:
            raise AssertionError("connection factory has no queued connection")
        return self.connections.popleft()


def _gr00t_actions(batch: int = 1, horizon: int = 40) -> dict[str, np.ndarray]:
    return {
        "eef_9d": np.ones((batch, horizon, 9), dtype=np.float32),
        "gripper_position": np.zeros((batch, horizon, 1), dtype=np.float32),
        "joint_position": np.full((batch, horizon, 7), 0.5, dtype=np.float32),
    }


@async_test
async def test_endpoint_handshake_inference_and_structural_protocol() -> None:
    codec = _PickleCodec()
    actions = _gr00t_actions()
    connection = _FakeConnection(
        [
            {"model": "nvidia/GR00T-N1.7-3B", "action_horizon": 40},
            actions,
        ],
        codec,
    )
    factory = _ConnectionFactory([connection])
    endpoint = OpenPiWebSocketEndpoint(
        "ws://policy.example/v1/realtime/robot/openpi",
        session_id="episode-7",
        expected_action_horizon=40,
        expected_action_dim=17,
        connection_factory=factory,
        codec=codec,
    )

    assert isinstance(endpoint, AsyncRemoteInferenceEndpoint)
    assert not endpoint.connected
    assert endpoint.connection_epoch == 0

    metadata = await endpoint.connect()
    assert metadata["model"] == "nvidia/GR00T-N1.7-3B"
    assert endpoint.connected
    assert endpoint.connection_epoch == 1

    result = await endpoint.infer_async(
        InferenceRequest(
            request_id="request-1",
            payload=RawRequest(
                observation={
                    "video": {"wrist": np.zeros((1, 2, 8, 8, 3), dtype=np.uint8)},
                    "state": {"joint_position": np.zeros((1, 1, 7), dtype=np.float32)},
                },
                prompt="pick up the object",
            ),
        )
    )

    assert result.request_id == "request-1"
    assert result.output["actions"].keys() == actions.keys()
    assert result.output["actions"]["eef_9d"].dtype == np.float32
    assert result.metadata["session_id"] == "episode-7"
    assert result.metadata["connection_epoch"] == 1
    assert result.metadata["server_metadata"]["action_horizon"] == 40
    assert result.execution_time_s >= 0
    assert len(connection.sent) == 1
    sent = connection.sent[0]
    assert sent["prompt"] == "pick up the object"
    assert sent["endpoint"] == "infer"
    assert sent["session_id"] == "episode-7"
    np.testing.assert_array_equal(
        sent["video"]["wrist"],
        np.zeros((1, 2, 8, 8, 3), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        sent["state"]["joint_position"],
        np.zeros((1, 1, 7), dtype=np.float32),
    )

    await endpoint.aclose()
    assert connection.closed
    assert not endpoint.connected
    assert endpoint.connection_epoch == 2


@async_test
async def test_reset_uses_old_session_then_rotates_it() -> None:
    codec = _PickleCodec()
    connection = _FakeConnection(
        [
            {"model": "test-policy"},
            {"status": "reset successful"},
            {"actions": np.ones((4, 3), dtype=np.float32), "server_steps": 6},
        ],
        codec,
    )
    endpoint = OpenPiWebSocketEndpoint(
        "ws://policy.example/openpi",
        session_id="old-session",
        expected_action_horizon=4,
        expected_action_dim=3,
        connection_factory=_ConnectionFactory([connection]),
        codec=codec,
    )

    status = await endpoint.reset({"reason": "episode_boundary"}, session_id="new-session")
    result = await endpoint.infer_async(
        InferenceRequest(payload={"observation/state": np.zeros(3, dtype=np.float32)})
    )

    assert status == "reset successful"
    assert endpoint.session_id == "new-session"
    assert connection.sent[0] == {
        "reason": "episode_boundary",
        "endpoint": "reset",
        "session_id": "old-session",
    }
    assert connection.sent[1]["endpoint"] == "infer"
    assert connection.sent[1]["session_id"] == "new-session"
    assert result.output["actions"].shape == (4, 3)
    assert result.metadata["response_metadata"] == {"server_steps": 6}


@async_test
async def test_transport_failure_reconnects_and_advances_epoch() -> None:
    codec = _PickleCodec()
    broken = _FakeConnection(
        [{"connection": "old"}],
        codec,
        receive_error_at=2,
    )
    recovered = _FakeConnection(
        [
            {"connection": "new"},
            np.ones((1, 8, 4), dtype=np.float32),
        ],
        codec,
    )
    factory = _ConnectionFactory([broken, recovered])
    endpoint = OpenPiWebSocketEndpoint(
        "wss://policy.example/openpi",
        expected_action_horizon=8,
        expected_action_dim=4,
        reconnect_attempts=1,
        connection_factory=factory,
        codec=codec,
    )

    result = await endpoint.infer_async(InferenceRequest(payload={"state": [1, 2, 3]}))

    assert result.output["actions"].shape == (1, 8, 4)
    assert result.metadata["server_metadata"] == {"connection": "new"}
    assert result.metadata["connection_epoch"] == 3
    assert len(factory.calls) == 2
    assert broken.closed
    assert endpoint.connected
    assert endpoint.connection_epoch == 3


@async_test
async def test_receive_timeout_disconnects_endpoint() -> None:
    codec = _PickleCodec()
    connection = _FakeConnection(
        [
            {"model": "slow-policy"},
            np.ones((2, 3), dtype=np.float32),
        ],
        codec,
        response_delay_s=0.05,
    )
    endpoint = OpenPiWebSocketEndpoint(
        "ws://policy.example/openpi",
        timeout_s=0.01,
        reconnect_attempts=0,
        connection_factory=_ConnectionFactory([connection]),
        codec=codec,
    )

    with pytest.raises(OpenPITimeoutError, match="timed out"):
        await endpoint.infer_async(InferenceRequest(payload={"state": [0]}))

    assert not endpoint.connected
    assert connection.closed
    assert endpoint.connection_epoch == 2


@pytest.mark.parametrize(
    "actions,match",
    [
        (
            np.zeros((1, 4, 2), dtype=np.float64),
            "dtype float32",
        ),
        (
            np.full((1, 4, 2), np.nan, dtype=np.float32),
            "non-finite",
        ),
        (
            {
                "arm": np.zeros((1, 4, 2), dtype=np.float32),
                "gripper": np.zeros((1, 5, 1), dtype=np.float32),
            },
            "share batch and horizon",
        ),
        (
            np.zeros((4,), dtype=np.float32),
            "must have shape",
        ),
    ],
)
def test_action_validation_rejects_unsafe_outputs(actions: Any, match: str) -> None:
    with pytest.raises(OpenPIProtocolError, match=match):
        validate_openpi_actions(actions)


def test_action_validation_accepts_unbatched_and_batched_common_shapes() -> None:
    unbatched = np.zeros((40, 17), dtype=np.float32)
    assert (
        validate_openpi_actions(
            unbatched,
            expected_batch_size=1,
            expected_action_horizon=40,
            expected_action_dim=17,
        )
        is unbatched
    )

    batched = _gr00t_actions(batch=2)
    validated = validate_openpi_actions(
        batched,
        expected_batch_size=2,
        expected_action_horizon=40,
        expected_action_dim=17,
    )
    assert validated.keys() == batched.keys()


def test_endpoint_validates_configuration_without_loading_optional_dependencies() -> None:
    endpoint = OpenPiWebSocketEndpoint("ws://configured-at-runtime.example/openpi")
    assert endpoint.uri == "ws://configured-at-runtime.example/openpi"

    with pytest.raises(ValueError, match="absolute ws"):
        OpenPiWebSocketEndpoint("127.0.0.1:8000")
    with pytest.raises(ValueError, match="timeout_s"):
        OpenPiWebSocketEndpoint("ws://policy.example/openpi", timeout_s=0)
    with pytest.raises(ValueError, match="reconnect_attempts"):
        OpenPiWebSocketEndpoint("ws://policy.example/openpi", reconnect_attempts=-1)


@pytest.mark.parametrize(
    "url",
    [
        "ws://localhost:8000/openpi",
        "ws://LOCALHOST.:8000/openpi",
        "ws://127.0.0.1:8000/openpi",
        "ws://127.42.0.9:8000/openpi",
        "ws://[::1]:8000/openpi",
    ],
)
def test_default_websocket_options_disable_proxy_for_loopback(url: str) -> None:
    assert _websocket_connect_options(url, 3.0)["proxy"] is None


def test_default_websocket_options_preserve_proxy_discovery_for_remote_host() -> None:
    options = _websocket_connect_options(
        "wss://remote-policy.example/openpi",
        3.0,
    )
    assert "proxy" not in options
