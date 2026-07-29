from __future__ import annotations

import asyncio
import importlib.util
import socket
from functools import wraps
from pathlib import Path
from types import ModuleType

from embodied_runtime.distributed.communication import (
    read_json_message,
    write_json_message,
)


def _load_client() -> ModuleType:
    path = Path(__file__).parents[2] / "examples" / "wifi_edge_client_py310.py"
    spec = importlib.util.spec_from_file_location("wifi_edge_client_py310", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def async_test(function):
    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


@async_test
async def test_standalone_client_runs_edge_then_blends_tcp_cloud_result() -> None:
    client = _load_client()
    observed_request: dict[str, object] = {}
    failure_socket = socket.socket()
    failure_socket.bind(("127.0.0.1", 0))
    failure_port = int(failure_socket.getsockname()[1])

    async def infer(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            observed_request.update(await read_json_message(reader))
            await asyncio.sleep(0.05)
            await write_json_message(
                writer,
                {
                    "ok": True,
                    "kind": "inference_result",
                    "request_id": observed_request["request_id"],
                    "actions": [[0.75, 0.75, 0.75] for _ in range(4)],
                    "action_shape": [4, 3],
                    "model_id": "fake-cloud-policy",
                    "device": "cuda:0",
                    "execution_time_s": 0.01,
                    "hostname": "fake-cloud",
                },
            )
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(infer, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    try:
        summary = await client.run_wifi_edge_client(
            client.ClientConfig(
                cloud_host="127.0.0.1",
                cloud_port=port,
                edge_device="cpu",
                cloud_weight=0.25,
                timeout_s=2.0,
                action_horizon=4,
                action_dim=3,
                observation_dim=5,
                hidden_dim=8,
                failure_probe_port=failure_port,
            )
        )
    finally:
        server.close()
        await server.wait_closed()
        failure_socket.close()

    assert observed_request["kind"] == "infer"
    assert observed_request["num_steps"] == 1
    assert summary["first_tick_source"] == "edge"
    assert summary["cloud_in_flight_after_first"] is True
    assert summary["second_tick_source"] == "blended"
    assert summary["action_shape"] == [4, 3]
    assert summary["final_action_device"] == "cpu"
    assert summary["cloud_model_id"] == "fake-cloud-policy"
    assert summary["cloud_weight"] == 0.25
    assert summary["fusion_max_abs_error"] < 1e-6
    assert summary["fused_vs_edge_max_abs"] > 0.0
    assert summary["failure_probe_tick_source"] == "edge"
    assert summary["failure_probe_reason"] == "cloud_error"
    assert summary["failure_probe_cloud_succeeded"] is False
    assert summary["failure_probe_action_shape"] == [4, 3]
    assert summary["failure_probe_action_device"] == "cpu"


@async_test
async def test_standalone_client_falls_back_to_edge_when_connection_is_refused() -> None:
    client = _load_client()
    unavailable_socket = socket.socket()
    unavailable_socket.bind(("127.0.0.1", 0))
    unavailable_port = int(unavailable_socket.getsockname()[1])
    try:
        summary = await client.run_wifi_edge_client(
            client.ClientConfig(
                cloud_host="127.0.0.1",
                cloud_port=unavailable_port,
                edge_device="cpu",
                timeout_s=0.5,
                action_horizon=4,
                action_dim=3,
                observation_dim=5,
                hidden_dim=8,
            )
        )
    finally:
        unavailable_socket.close()

    assert summary["first_tick_source"] == "edge"
    assert summary["second_tick_source"] == "edge"
    assert summary["cloud_request_succeeded"] is False
    assert summary["fallback_reason"] == "cloud_error"
    assert summary["cloud_error_type"] in {"ConnectionRefusedError", "OSError"}
    assert summary["action_shape"] == [4, 3]
    assert summary["final_action_device"] == "cpu"
    assert summary["fusion_max_abs_error"] is None
