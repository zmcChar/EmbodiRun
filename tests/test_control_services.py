"""Boundary tests for Host-to-Control contracts and control-node execution."""

from __future__ import annotations

import socket
from dataclasses import replace
from threading import Thread
from types import SimpleNamespace

import pytest

from rlinf_deploy.robots.sensors import SensorInput
from rlinf_deploy.services.control.contracts import (
    ControlContractError,
    ControlServiceConfig,
    TaskRequest,
    TaskResult,
)
from rlinf_deploy.services.control.server import (
    ControlHttpServer,
    ControlService,
    ControlServiceError,
)
from rlinf_deploy.services.host.control import ControlClient
from rlinf_deploy.services.host.executor import LocalExecutor


def control_config(*, port: int = 8100) -> ControlServiceConfig:
    return ControlServiceConfig(
        runtime_id="so101-runtime",
        binding_kind="lerobot.so101.pi05",
        bind="127.0.0.1",
        port=port,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:8000",
        inference_options={},
        robot_id="so101-1",
        robot_kind="lerobot.so101",
        robot_options={"port": "/dev/ttyACM0"},
        inputs=(
            SensorInput(
                "front-camera",
                "observation.images.front",
                "v4l2",
                {"device": "/dev/video0"},
            ),
        ),
        runtime_options={},
    )


def task_request() -> TaskRequest:
    return TaskRequest(
        request_id="task-1",
        runtime_id="so101-runtime",
        prompt="抓取黄色格子",
        chunk_steps=10,
        max_steps=2,
        control_hz=5.0,
        inference_timeout_s=60.0,
    )


def test_host_control_contract_contains_only_dynamic_task_fields() -> None:
    payload = task_request().to_payload()

    assert TaskRequest.from_payload(payload) == task_request()
    assert payload["schema"] == "rlinf.control.task.v1"
    assert "robot" not in payload
    assert "inputs" not in payload
    assert "binding" not in payload
    assert "inference" not in payload


def test_host_control_contract_rejects_unknown_fields() -> None:
    payload = task_request().to_payload()
    payload["robot"] = {"id": "must-not-cross-this-boundary"}

    with pytest.raises(ControlContractError, match="unknown fields: robot"):
        TaskRequest.from_payload(payload)


def test_control_service_config_round_trip_keeps_static_node_ownership() -> None:
    config = control_config()

    decoded = ControlServiceConfig.from_json(config.to_json())

    assert decoded == config
    assert decoded.robot_options == {"port": "/dev/ttyACM0"}
    assert decoded.inputs[0].sensor_id == "front-camera"


def test_control_service_rejects_non_loopback_task_listener() -> None:
    with pytest.raises(ValueError, match="loopback"):
        replace(control_config(), bind="0.0.0.0")


def test_host_client_rejects_non_loopback_control_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        ControlClient(LocalExecutor(), "http://192.168.2.232:8100")


def test_control_service_executes_chunks_and_releases_resources(monkeypatch) -> None:
    events: list[object] = []

    class Client:
        def health(self):
            return {"status": "ok"}

        def shutdown(self):
            events.append("client.shutdown")

    class Cameras:
        def capture(self):
            events.append("camera.capture")
            return ()

        def close(self):
            events.append("camera.close")

    class Robot:
        def __init__(self, config):
            events.append(("robot.config", config))

        def connect(self):
            events.append("robot.connect")

        def close(self):
            events.append("robot.close")

    class Runtime:
        def __init__(self, robot, client, **options):
            events.append(("runtime.options", options))

        def step(self, frames):
            events.append(("runtime.step", frames))

        def close(self):
            events.append("runtime.close")

    binding = SimpleNamespace(
        kind="lerobot.so101.pi05",
        maximum_chunk_steps=50,
        mapper_factory=lambda: "mapper",
    )
    robot = SimpleNamespace(
        config_factory=lambda robot_id, options: (robot_id, dict(options)),
        adapter_type=Robot,
    )
    monkeypatch.setattr(
        "rlinf_deploy.services.control.server._definitions",
        lambda _config: (binding, robot),
    )
    service = ControlService(
        control_config(),
        camera_factory=lambda _inputs: Cameras(),
        client_factory=lambda _config, _timeout: Client(),
        runtime_factory=Runtime,
    )

    result = service.execute(task_request())

    assert result == TaskResult("task-1", "so101-runtime", 2)
    assert events.count("camera.capture") == 2
    assert events[-4:] == [
        "runtime.close",
        "robot.close",
        "camera.close",
        "client.shutdown",
    ]
    runtime_options = next(
        value
        for item in events
        if isinstance(item, tuple) and item[0] == "runtime.options"
        for value in (item[1],)
    )
    assert runtime_options["instruction"] == "抓取黄色格子"
    assert runtime_options["chunk_steps"] == 10
    assert runtime_options["control_hz"] == 5.0


def test_control_service_serializes_robot_tasks() -> None:
    service = ControlService(control_config())
    service._task_lock.acquire()
    try:
        with pytest.raises(ControlServiceError, match="already executing"):
            service.execute(task_request())
    finally:
        service._task_lock.release()


def test_host_client_submits_task_to_control_http_service() -> None:
    port = _unused_loopback_port()
    config = control_config(port=port)

    class Service:
        def __init__(self):
            self.config = config
            self.received = None

        def health(self):
            return {"status": "ok", "runtime_id": config.runtime_id}

        def execute(self, request):
            self.received = request
            return TaskResult(request.request_id, request.runtime_id, 2)

    service = Service()
    server = ControlHttpServer(service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = ControlClient(
            LocalExecutor(), f"http://127.0.0.1:{port}"
        ).run(task_request(), timeout_s=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert result == TaskResult("task-1", "so101-runtime", 2)
    assert service.received == task_request()
    assert not thread.is_alive()


def _unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
