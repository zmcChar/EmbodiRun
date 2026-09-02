"""Run one bounded SO-101 binding worker on its deployment node."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rlinf_deploy.bindings import BindingMapper, binding_definition
from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter, SO101Config
from rlinf_deploy.robots.sensors import SensorInput
from rlinf_deploy.robots.sensors.cameras import (
    CameraSource,
    create_camera_source,
)

from .request import SO101WorkerRequest
from .runtime import SO101Runtime


class WorkerExecutionError(RuntimeError):
    """The configured SO-101 worker cannot execute safely."""


def execute(
    request: SO101WorkerRequest,
    *,
    camera_factory: Callable[[Sequence[SensorInput]], CameraSource] = (
        create_camera_source
    ),
    robot_factory: Callable[[SO101Config], Any] = SO101Adapter,
    client_factory: Callable[..., Any] = VvlaHttpClient,
    runtime_factory: Callable[..., Any] = SO101Runtime,
    emit: Callable[[Mapping[str, object]], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run at most ``max_steps`` actions and always release cameras and motors."""

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

    mapper = _binding_mapper(request.binding_kind)
    cameras = camera_factory(request.inputs)
    robot: Any | None = None
    controller: Any | None = None
    completed = 0
    try:
        robot = robot_factory(request.robot)
        robot.connect()
        controller = runtime_factory(
            robot,
            client,
            instruction=request.prompt,
            mapper=mapper,
        )
        period_s = 1.0 / request.control_hz
        for _ in range(request.max_steps):
            started_s = monotonic()
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
            remaining_s = period_s - (monotonic() - started_s)
            if remaining_s > 0 and completed < request.max_steps:
                sleep(remaining_s)
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
    parser = argparse.ArgumentParser(prog="rlinf-so101-worker")
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    try:
        execute(SO101WorkerRequest.from_json(args.request_json))
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


def _binding_mapper(binding_kind: str) -> BindingMapper:
    try:
        definition = binding_definition(binding_kind)
    except (KeyError, TypeError):
        raise WorkerExecutionError(
            f"binding {binding_kind!r} is not available"
        ) from None
    if definition.robot_kind != "lerobot.so101":
        raise WorkerExecutionError(
            f"binding {binding_kind!r} does not target lerobot.so101"
        )
    if definition.mapper_factory is None:
        raise WorkerExecutionError(
            f"binding {binding_kind!r} does not provide a mapper"
        )
    return definition.mapper_factory()


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SO101WorkerRequest",
    "WorkerExecutionError",
    "execute",
    "main",
]
