"""Serve Host tasks on a control node and execute its configured robot runtime."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rlinf_deploy.bindings import BindingDefinition, binding_definition
from rlinf_deploy.robots import RobotDefinition, robot_definition
from rlinf_deploy.robots.sensors import SensorInput
from rlinf_deploy.robots.sensors.cameras import CameraSource, create_camera_source
from rlinf_deploy.services.inference import build_inference_client

from .contracts import (
    ControlContractError,
    ControlServiceConfig,
    TaskRequest,
    TaskResult,
    error_payload,
)
from .runtime import ControlRuntime

_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_REQUEST_BYTES = 64 * 1024


class ControlServiceError(RuntimeError):
    """A configured control service cannot execute a task safely."""


class ControlTaskRejected(ControlServiceError):
    """A valid task conflicts with this runtime or its current state."""


class ControlService:
    """Validate, serialize, and execute tasks for exactly one robot runtime."""

    def __init__(
        self,
        config: ControlServiceConfig,
        *,
        camera_factory: Callable[[Sequence[SensorInput]], CameraSource] = (
            create_camera_source
        ),
        client_factory: Callable[[ControlServiceConfig, float], Any] | None = None,
        runtime_factory: Callable[..., Any] = ControlRuntime,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.camera_factory = camera_factory
        self.client_factory = client_factory or create_inference_client
        self.runtime_factory = runtime_factory
        self.monotonic = monotonic
        self.sleep = sleep
        self._task_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._wireless_client: Any | None = None

    def health(self) -> dict[str, Any]:
        """Report ready only when the configured inference service is reachable."""

        client = self._inference_client(2.0)
        try:
            health = client.health()
        finally:
            self._release_inference_client(client)
        if health.get("status") != "ok":
            raise ControlServiceError(
                f"inference service at {self.config.inference_endpoint} is not healthy"
            )
        return {"status": "ok", "runtime_id": self.config.runtime_id}

    def execute(self, request: TaskRequest) -> TaskResult:
        """Run one exclusive task and release all hardware and session resources."""

        if request.runtime_id != self.config.runtime_id:
            raise ControlTaskRejected(
                f"runtime {request.runtime_id!r} is not served by this control service"
            )
        if not self._task_lock.acquire(blocking=False):
            raise ControlTaskRejected("control service is already executing a task")
        try:
            return self._execute_locked(request)
        finally:
            self._task_lock.release()

    def _execute_locked(self, request: TaskRequest) -> TaskResult:
        binding, robot_definition_value = _definitions(self.config)
        if request.chunk_steps > binding.maximum_chunk_steps:
            raise ControlTaskRejected(
                f"chunk_steps {request.chunk_steps} exceeds binding "
                f"{binding.kind!r} maximum {binding.maximum_chunk_steps}"
            )
        if self.config.runtime_options:
            names = ", ".join(sorted(self.config.runtime_options))
            raise ControlServiceError(f"unsupported runtime options: {names}")
        try:
            robot_config = robot_definition_value.config_factory(
                self.config.robot_id,
                self.config.robot_options,
            )
        except (TypeError, ValueError) as error:
            raise ControlServiceError(
                f"robot {self.config.robot_id!r} configuration is invalid: {error}"
            ) from error

        client = self._inference_client(request.inference_timeout_s)
        cameras: CameraSource | None = None
        robot: Any | None = None
        controller: Any | None = None
        completed = 0
        try:
            health = client.health()
            if health.get("status") != "ok":
                raise ControlServiceError(
                    f"inference service at {self.config.inference_endpoint} "
                    "is not healthy"
                )
            cameras = self.camera_factory(self.config.inputs)
            robot = robot_definition_value.adapter_type(robot_config)
            robot.connect()
            controller = self.runtime_factory(
                robot,
                client,
                instruction=request.prompt,
                mapper=binding.mapper_factory(),
                chunk_steps=request.chunk_steps,
                control_hz=request.control_hz,
                monotonic=self.monotonic,
                sleep=self.sleep,
            )
            for _ in range(request.max_steps):
                controller.step(cameras.capture())
                completed += 1
        finally:
            try:
                if controller is not None:
                    controller.close()
            finally:
                try:
                    if robot is not None:
                        robot.close()
                finally:
                    try:
                        if cameras is not None:
                            cameras.close()
                    finally:
                        self._release_inference_client(client)
        return TaskResult(request.request_id, request.runtime_id, completed)

    def close(self) -> None:
        """Release the process-owned wireless connection, when configured."""

        with self._client_lock:
            client = self._wireless_client
            self._wireless_client = None
        if client is not None:
            _shutdown_client(client)

    def _inference_client(self, timeout_s: float) -> Any:
        if self.config.inference_transport != "wireless":
            return self.client_factory(self.config, timeout_s)
        with self._client_lock:
            if self._wireless_client is None:
                self._wireless_client = self.client_factory(self.config, timeout_s)
            client = self._wireless_client
        with_timeout = getattr(client, "with_timeout", None)
        return with_timeout(timeout_s) if callable(with_timeout) else client

    def _release_inference_client(self, client: Any) -> None:
        if self.config.inference_transport != "wireless":
            _shutdown_client(client)


def create_inference_client(config: ControlServiceConfig, timeout_s: float) -> Any:
    """Select the Control-to-Inference client from the static runtime config."""

    return build_inference_client(
        config.inference_transport,
        config.inference_endpoint,
        config.inference_options,
        backend=config.inference_backend,
        timeout_s=timeout_s,
    )


class ControlHttpServer(ThreadingHTTPServer):
    """HTTP listener carrying one control service on a loopback-only socket."""

    daemon_threads = True

    def __init__(self, service: ControlService) -> None:
        self.control_service = service
        if ":" in service.config.bind:
            self.address_family = socket.AF_INET6
        super().__init__(
            (service.config.bind, service.config.port),
            ControlRequestHandler,
        )


class ControlRequestHandler(BaseHTTPRequestHandler):
    """Small JSON HTTP surface for health checks and task submission."""

    server: ControlHttpServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/healthz":
            self._send(HTTPStatus.NOT_FOUND, error_payload("route not found"))
            return
        try:
            payload = self.server.control_service.health()
        except Exception as error:  # noqa: BLE001 - normalize service failures
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "starting", "error": str(error)},
            )
            return
        self._send(HTTPStatus.OK, payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/v1/tasks":
            self._send(HTTPStatus.NOT_FOUND, error_payload("route not found"))
            return
        try:
            request = TaskRequest.from_payload(self._read_json())
        except (ControlContractError, OSError, ValueError) as error:
            self._send(HTTPStatus.BAD_REQUEST, error_payload(str(error)))
            return
        try:
            result = self.server.control_service.execute(request)
        except ControlTaskRejected as error:
            status = (
                HTTPStatus.CONFLICT
                if "already executing" in str(error)
                else HTTPStatus.BAD_REQUEST
            )
            self._send(status, error_payload(str(error)))
            return
        except Exception as error:  # noqa: BLE001 - normalize service failures
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, error_payload(str(error)))
            return
        self._send(HTTPStatus.OK, result.to_payload())

    def _read_json(self) -> object:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        length_value = self.headers.get("Content-Length")
        try:
            length = int(length_value or "")
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        if not 0 < length <= _MAX_REQUEST_BYTES:
            raise ValueError(
                f"request body must be between 1 and {_MAX_REQUEST_BYTES} bytes"
            )
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid UTF-8 JSON") from error

    def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Load one generated runtime and serve it until the supervisor stops us."""

    parser = argparse.ArgumentParser(prog="rlinf-control-serve")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        content = args.config.read_bytes()
        if len(content) > _MAX_CONFIG_BYTES:
            raise ValueError("control config is too large")
        config = ControlServiceConfig.from_json(content.decode("utf-8"))
        server = ControlHttpServer(ControlService(config))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        parser.error(str(error))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.control_service.close()
    return 0


def _definitions(
    config: ControlServiceConfig,
) -> tuple[BindingDefinition, RobotDefinition]:
    try:
        binding = binding_definition(config.binding_kind)
    except (KeyError, TypeError):
        raise ControlServiceError(
            f"binding {config.binding_kind!r} is not available"
        ) from None
    if binding.robot_kind != config.robot_kind:
        raise ControlServiceError(
            f"binding {config.binding_kind!r} does not target {config.robot_kind!r}"
        )
    try:
        robot = robot_definition(config.robot_kind)
    except (KeyError, TypeError):
        raise ControlServiceError(
            f"robot type {config.robot_kind!r} is not available"
        ) from None
    return binding, robot


def _shutdown_client(client: Any) -> None:
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ControlHttpServer",
    "ControlService",
    "ControlServiceError",
    "create_inference_client",
    "main",
]
