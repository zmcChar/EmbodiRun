"""Generic process entry point for a configured robot-policy binding."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots import RobotDefinition, robot_definition
from rlinf_deploy.robots.sensors import SensorInput
from rlinf_deploy.robots.sensors.cameras import (
    CameraSource,
    create_camera_source,
)

from . import BindingDefinition, binding_definition
from .request import BindingWorkerRequest
from .runtime import BindingRuntime


class WorkerExecutionError(RuntimeError):
    """The configured binding worker cannot execute safely."""


def execute(
    request: BindingWorkerRequest,
    *,
    camera_factory: Callable[[Sequence[SensorInput]], CameraSource] = (
        create_camera_source
    ),
    client_factory: Callable[..., Any] = VvlaHttpClient,
    runtime_factory: Callable[..., Any] = BindingRuntime,
    emit: Callable[[Mapping[str, object]], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run a bounded control loop and always release cameras and hardware."""

    binding, robot_definition_value = _definitions(request)
    if request.runtime_options:
        names = ", ".join(sorted(request.runtime_options))
        raise WorkerExecutionError(f"unsupported runtime options: {names}")
    try:
        robot_config = robot_definition_value.config_factory(
            request.robot_id,
            request.robot_options,
        )
    except (TypeError, ValueError) as error:
        raise WorkerExecutionError(
            f"robot {request.robot_id!r} configuration is invalid: {error}"
        ) from error

    output = emit or _emit
    client = client_factory(
        request.model_endpoint,
        timeout_s=request.request_timeout_s,
    )
    health = client.health()
    if health.get("status") != "ok":
        raise WorkerExecutionError(
            f"model service at {request.model_endpoint} is not healthy"
        )

    mapper = binding.mapper_factory()
    cameras = camera_factory(request.inputs)
    robot: Any | None = None
    controller: Any | None = None
    completed = 0
    try:
        robot = robot_definition_value.adapter_type(robot_config)
        robot.connect()
        controller = runtime_factory(
            robot,
            client,
            instruction=request.prompt,
            mapper=mapper,
            control_hz=request.control_hz,
            monotonic=monotonic,
            sleep=sleep,
        )
        for _ in range(request.max_steps):
            result = controller.step(cameras.capture())
            completed += 1
            output(
                {
                    "event": "step",
                    "runtime": request.runtime_id,
                    "step": completed,
                    "session_revision": result.session_revision,
                }
            )
    finally:
        try:
            if controller is not None:
                controller.close()
        finally:
            try:
                if robot is not None:
                    robot.close()
            finally:
                cameras.close()
    output(
        {
            "event": "complete",
            "runtime": request.runtime_id,
            "steps": completed,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rlinf-binding-worker")
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    try:
        execute(BindingWorkerRequest.from_json(args.request_json))
    except (OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {"event": "error", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    return 0


def _definitions(
    request: BindingWorkerRequest,
) -> tuple[BindingDefinition, RobotDefinition]:
    try:
        binding = binding_definition(request.binding_kind)
    except (KeyError, TypeError):
        raise WorkerExecutionError(
            f"binding {request.binding_kind!r} is not available"
        ) from None
    if binding.robot_kind != request.robot_kind:
        raise WorkerExecutionError(
            f"binding {request.binding_kind!r} does not target "
            f"{request.robot_kind!r}"
        )
    try:
        robot = robot_definition(request.robot_kind)
    except (KeyError, TypeError):
        raise WorkerExecutionError(
            f"robot type {request.robot_kind!r} is not available"
        ) from None
    return binding, robot


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BindingWorkerRequest",
    "WorkerExecutionError",
    "execute",
    "main",
]
