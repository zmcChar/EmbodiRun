from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest

from embodied_runtime.apps.cloud_edge.pi05_codec import pi05_response_to_result
from embodied_runtime.apps.cloud_edge.smolvla_runtime import (
    EdgeRuntime,
    Pi05TcpEndpoint,
    build_edge_runtime,
)
from embodied_runtime.apps.cloud_edge.smolvla_scenario import run_async_scenario
from embodied_runtime.apps.cloud_edge.smolvla_settings import SmolVLAPi05AsyncConfig
from embodied_runtime.distributed import FailoverConfig, FailoverMode
from embodied_runtime.engine import InferenceResult
from embodied_runtime.models.plans import IterativeFlowPlan


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class _FakeActions:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeEdgeEngine:
    def __init__(self, *, delay_s: float = 0.001) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.started = False
        self.closed = False
        self.package = SimpleNamespace(
            spec=SimpleNamespace(
                model_id="fake-smolvla",
                action_horizon=50,
                action_dim=6,
            )
        )

    async def start(self) -> None:
        self.started = True

    async def infer_async(self, request) -> InferenceResult:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return InferenceResult(
            request_id=request.request_id,
            output={"actions": _FakeActions((50, 6))},
            execution_time_s=self.delay_s,
            metadata={"model_id": "fake-smolvla", "device_id": "cuda:0"},
        )

    async def aclose(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _FakeCloudEndpoint:
    def __init__(self, *, delay_s: float = 0.02) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self._connected = True
        self._connection_epoch = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    def set_connected(self, connected: bool) -> None:
        if connected != self._connected:
            self._connected = connected
            self._connection_epoch += 1

    async def infer_async(self, request) -> InferenceResult:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return InferenceResult(
            request_id=request.request_id,
            output={"actions": _FakeActions((50, 32))},
            execution_time_s=self.delay_s,
            metadata={"model_id": "fake-pi05", "device_id": "cuda:0"},
        )


def _config(**changes: Any) -> SmolVLAPi05AsyncConfig:
    values = {
        "edge_checkpoint": "/models/smolvla",
        "edge_vlm_base_path": "/models/smolvlm2",
        "cloud_host": "cloud.example.invalid",
        "failover": FailoverConfig(
            mode=FailoverMode.ASYNC_CLOUD_PREFERRED,
            cloud_request_timeout_s=1.0,
            cloud_result_ttl_s=1.0,
            max_cloud_sequence_lag=4,
            cloud_submit_interval_s=60.0,
        ),
    }
    values.update(changes)
    return SmolVLAPi05AsyncConfig(**values)


def _runtime(engine: _FakeEdgeEngine) -> EdgeRuntime:
    return EdgeRuntime(
        adapter=object(),
        engine=engine,
        session=SimpleNamespace(device=SimpleNamespace(device_id="cuda:0")),
        payload={"synthetic": _FakeActions((1, 1))},
        load_time_s=0.5,
    )


@async_test
async def test_first_tick_does_not_wait_for_cloud_then_cache_preempts_and_disconnects() -> None:
    engine = _FakeEdgeEngine()
    cloud = _FakeCloudEndpoint()
    runtime = _runtime(engine)

    summary = await run_async_scenario(runtime, cloud, _config())

    assert summary["first_tick"]["source"] == "edge"
    assert summary["first_tick"]["selected_action_shape"] == [50, 6]
    assert summary["cloud_in_flight_after_first"] is True
    assert summary["first_tick"]["control_path_time_s"] < cloud.delay_s
    assert summary["second_tick"]["source"] == "cloud"
    assert summary["second_tick"]["source_sequence_id"] == 1
    assert summary["second_tick"]["selected_action_shape"] == [50, 32]
    assert summary["disconnected_tick"]["source"] == "edge"
    assert summary["disconnected_tick"]["fallback_reason"] == "cloud_disconnected"
    assert summary["disconnected_tick"]["selected_action_shape"] == [50, 6]
    assert summary["numeric_blend_performed"] is False
    assert summary["action_contracts"]["semantically_compatible"] is False
    assert engine.calls == 3
    assert cloud.calls == 1
    assert engine.started
    assert engine.closed
    assert runtime._closed


@async_test
async def test_edge_only_never_calls_tcp_endpoint() -> None:
    engine = _FakeEdgeEngine()
    cloud = _FakeCloudEndpoint()
    runtime = _runtime(engine)
    config = replace(
        _config(),
        failover=FailoverConfig(mode=FailoverMode.EDGE_ONLY),
    )

    summary = await run_async_scenario(runtime, cloud, config)

    assert cloud.calls == 0
    assert summary["cloud_in_flight_after_first"] is False
    assert summary["first_tick"]["source"] == "edge"
    assert summary["second_tick"]["source"] == "edge"
    assert summary["disconnected_tick"]["source"] == "edge"
    assert summary["numeric_blend_performed"] is False


def test_async_blend_is_rejected_before_models_or_network_are_touched() -> None:
    with pytest.raises(ValueError, match="async_blend"):
        _config(failover=FailoverConfig(mode=FailoverMode.ASYNC_BLEND))


class _FakeJsonClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def request(self, payload):
        self.requests.append(dict(payload))
        return self.response


@async_test
async def test_tcp_wrapper_validates_response_and_observes_logical_disconnect() -> None:
    response = {
        "ok": True,
        "kind": "inference_result",
        "request_id": "cloud-1",
        "actions": [[0.25] * 32 for _ in range(50)],
        "action_shape": [50, 32],
        "model_id": "pi05",
        "device": "cuda:0",
        "execution_time_s": 0.2,
    }
    client = _FakeJsonClient(response)
    endpoint = Pi05TcpEndpoint(client)
    result = await endpoint.infer_async(SimpleNamespace(request_id="cloud-1", num_steps=10, seed=7))

    assert result.request_id == "cloud-1"
    assert result.output["actions"] == response["actions"]
    assert result.metadata["action_shape"] == (50, 32)
    assert client.requests == [
        {
            "kind": "infer",
            "request_id": "cloud-1",
            "num_steps": 10,
            "seed": 7,
        }
    ]

    endpoint.set_connected(False)
    assert endpoint.connection_epoch == 1
    with pytest.raises(ConnectionError, match="logically disconnected"):
        await endpoint.infer_async(SimpleNamespace(request_id="cloud-2", num_steps=10, seed=8))
    assert len(client.requests) == 1


def test_cloud_response_rejects_action_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="must have shape"):
        pi05_response_to_result(
            {
                "ok": True,
                "kind": "inference_result",
                "request_id": "cloud-1",
                "actions": [[0.0] * 31 for _ in range(50)],
            },
            "cloud-1",
        )


def test_edge_builder_uses_adapter_backend_engine_once_and_preserves_dtype() -> None:
    package = SimpleNamespace(
        plan=IterativeFlowPlan(),
        spec=SimpleNamespace(model_id="fake-smolvla"),
    )
    adapter = SimpleNamespace(
        collate=object(),
        unbatch=object(),
        build_calls=[],
        synthetic_calls=[],
    )

    def build_package(checkpoint, **options):
        adapter.build_calls.append((checkpoint, options))
        return package

    def synthetic_batch(**options):
        adapter.synthetic_calls.append(options)
        return {"payload": _FakeActions((1, 1))}

    adapter.build_package = build_package
    adapter.synthetic_batch = synthetic_batch

    session = SimpleNamespace(
        closed=False,
        device=SimpleNamespace(device_id="cuda:0"),
    )
    session.close = lambda: setattr(session, "closed", True)
    compile_calls = []
    backend = SimpleNamespace(
        probe=lambda: (SimpleNamespace(device_id="cuda:0"),),
        compile=lambda package_arg, device_arg, options_arg: (
            compile_calls.append((package_arg, device_arg, options_arg)) or "artifact"
        ),
        load=lambda artifact: session,
    )
    engine_calls = []

    class FakeBuiltEngine:
        def __init__(self, *args, **kwargs) -> None:
            engine_calls.append((args, kwargs))
            self.closed = False

        def close(self) -> None:
            self.closed = True
            session.close()

    runtime = build_edge_runtime(
        _config(),
        adapter_factory=lambda: adapter,
        backend_factory=lambda: backend,
        engine_factory=FakeBuiltEngine,
    )
    try:
        assert adapter.build_calls == [
            (
                "/models/smolvla",
                {
                    "vlm_base_path": "/models/smolvlm2",
                    "stats_variant": "so100",
                    "local_files_only": True,
                },
            )
        ]
        assert adapter.synthetic_calls == [{"batch_size": 1, "language_length": 48, "seed": 0}]
        assert len(compile_calls) == 1
        assert compile_calls[0][2].dtype is None
        assert compile_calls[0][2].mode == "eager"
        assert len(engine_calls) == 1
        assert engine_calls[0][1] == {
            "batcher": adapter.collate,
            "splitter": adapter.unbatch,
        }
    finally:
        runtime.close()

    assert session.closed
