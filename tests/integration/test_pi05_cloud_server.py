from __future__ import annotations

import asyncio
import socket
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest

from embodied_runtime.apps.pi05_cloud_server import (
    Pi05CloudServerConfig,
    _build_cloud_runtime,
    _CloudRuntime,
    _dispatch,
    _serve,
)
from embodied_runtime.distributed.communication import TcpJsonRequestClient
from embodied_runtime.engine import InferenceResult


@pytest.fixture
def free_tcp_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class _FakeActions:
    shape = (1, 2)

    def detach(self) -> _FakeActions:
        return self

    def float(self) -> _FakeActions:
        return self

    def cpu(self) -> _FakeActions:
        return self

    def tolist(self) -> list[list[float]]:
        return [[1.0, 2.0]]


class _FakeEngine:
    def __init__(self) -> None:
        self.package = SimpleNamespace(spec=SimpleNamespace(model_id="fake-pi05"))
        self.session = SimpleNamespace(device=SimpleNamespace(device_id="cuda:0"))
        self.started = False
        self.closed = False
        self.last_request = None

    async def start(self) -> None:
        self.started = True

    async def aclose(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True

    async def infer_async(self, request) -> InferenceResult:
        self.last_request = request
        return InferenceResult(
            request_id=request.request_id,
            output={"actions": _FakeActions()},
            execution_time_s=0.25,
            metadata={"model_id": "fake-pi05", "device_id": "cuda:0"},
        )


def _config(**changes: Any) -> Pi05CloudServerConfig:
    values = {
        "checkpoint": "fake-checkpoint",
        "num_steps": 4,
        "host": "127.0.0.1",
        "port": 18765,
    }
    values.update(changes)
    return Pi05CloudServerConfig(**values)


@async_test
async def test_dispatches_ping_and_bounded_inference_without_real_model() -> None:
    engine = _FakeEngine()
    runtime = _CloudRuntime(adapter=object(), engine=engine, payload={"input": 1})
    config = _config()

    pong = await _dispatch(runtime, config, {"kind": "ping"})
    result = await _dispatch(
        runtime,
        config,
        {"kind": "infer", "request_id": "request-1", "num_steps": 2, "seed": 7},
    )

    assert pong["kind"] == "pong"
    assert result == {
        "ok": True,
        "kind": "inference_result",
        "request_id": "request-1",
        "actions": [[1.0, 2.0]],
        "action_shape": [1, 2],
        "model_id": "fake-pi05",
        "device": "cuda:0",
        "execution_time_s": 0.25,
        "hostname": result["hostname"],
    }
    assert engine.last_request.payload == {"input": 1}
    assert engine.last_request.num_steps == 2
    assert engine.last_request.seed == 7

    with pytest.raises(ValueError, match="exceeds configured server limit"):
        await _dispatch(
            runtime,
            config,
            {"kind": "infer", "request_id": "request-2", "num_steps": 5},
        )


@async_test
async def test_real_tcp_server_ping_and_shutdown_without_real_model(
    free_tcp_port: int,
) -> None:
    engine = _FakeEngine()
    runtime = _CloudRuntime(adapter=object(), engine=engine, payload={})
    config = _config(port=free_tcp_port)
    task = asyncio.create_task(_serve(runtime, config))

    client = TcpJsonRequestClient("127.0.0.1", free_tcp_port)
    try:
        for _ in range(100):
            try:
                response = await client.request({"kind": "ping"})
                break
            except (ConnectionError, OSError):
                await asyncio.sleep(0.01)
        else:
            pytest.fail("cloud server did not become ready")
        assert response["kind"] == "pong"
        assert response["device"] == "cuda:0"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert engine.started
    assert engine.closed
    assert runtime._closed


def test_build_runtime_closes_loaded_session_if_engine_construction_fails() -> None:
    session = SimpleNamespace(closed=False)

    def close() -> None:
        session.closed = True

    session.close = close
    package = SimpleNamespace(plan=SimpleNamespace(step="step"))
    adapter = SimpleNamespace(
        build_package=lambda *_args, **_kwargs: package,
        collate=object(),
        unbatch=object(),
    )
    backend = SimpleNamespace(
        probe=lambda: (SimpleNamespace(device_id="cuda:0"),),
        compile=lambda *_args, **_kwargs: object(),
        load=lambda _artifact: session,
    )

    def fail_engine(*_args, **_kwargs):
        raise RuntimeError("engine construction failed")

    with pytest.raises(RuntimeError, match="engine construction failed"):
        _build_cloud_runtime(
            _config(),
            adapter_factory=lambda: adapter,
            backend_factory=lambda: backend,
            engine_factory=fail_engine,
        )

    assert session.closed


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"checkpoint": ""}, "checkpoint"),
        ({"device": "cpu"}, "cuda"),
        ({"dtype": "int8"}, "dtype"),
        ({"host": ""}, "host"),
        ({"port": 0}, "port"),
        ({"request_timeout_s": float("nan")}, "request_timeout_s"),
    ],
)
def test_server_config_rejects_invalid_values(
    changes: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _config(**changes)
