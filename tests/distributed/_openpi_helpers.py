from __future__ import annotations

import asyncio
import pickle
import time
from collections import deque
from functools import wraps
from typing import Any

import numpy as np


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class PickleCodec:
    def pack(self, value: Any) -> bytes:
        return pickle.dumps(value)

    def unpack(self, value: bytes | str) -> Any:
        assert isinstance(value, bytes)
        return pickle.loads(value)


class FakeConnection:
    def __init__(
        self,
        responses: list[Any],
        codec: PickleCodec,
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


class ConnectionFactory:
    def __init__(self, connections: list[FakeConnection]) -> None:
        self.connections = deque(connections)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, uri: str, timeout_s: float) -> FakeConnection:
        self.calls.append((uri, timeout_s))
        if not self.connections:
            raise AssertionError("connection factory has no queued connection")
        return self.connections.popleft()


def gr00t_actions(batch: int = 1, horizon: int = 40) -> dict[str, np.ndarray]:
    return {
        "eef_9d": np.ones((batch, horizon, 9), dtype=np.float32),
        "gripper_position": np.zeros((batch, horizon, 1), dtype=np.float32),
        "joint_position": np.full((batch, horizon, 7), 0.5, dtype=np.float32),
    }
