"""Run one bounded SO-101/Pi0.5 worker on its deployment node."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter, SO101Config
from rlinf_deploy.robots.sensors.cameras import (
    CameraSource,
    V4L2CameraConfig,
    V4L2CameraSource,
)

from .runtime import Pi05SO101Runtime
from .spec import SO101Pi05Spec


class RuntimeExecutionError(RuntimeError):
    """The configured runtime cannot execute safely."""


def execute(
    spec: SO101Pi05Spec,
    *,
    camera_factory: Callable[
        [Sequence[V4L2CameraConfig]], CameraSource
    ] = V4L2CameraSource,
    robot_factory: Callable[[SO101Config], Any] = SO101Adapter,
    client_factory: Callable[..., Any] = VvlaHttpClient,
    runtime_factory: Callable[..., Any] = Pi05SO101Runtime,
    emit: Callable[[Mapping[str, object]], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run at most ``max_steps`` actions and always release cameras and motors."""

    output = emit or _emit
    client = client_factory(spec.model_endpoint, timeout_s=spec.request_timeout_s)
    health = client.health()
    if health.get("status") != "ok":
        raise RuntimeExecutionError(
            f"model service at {spec.model_endpoint} is not healthy"
        )

    cameras = camera_factory(spec.cameras)
    robot: Any | None = None
    controller: Any | None = None
    completed = 0
    try:
        robot = robot_factory(spec.robot)
        robot.connect()
        controller = runtime_factory(robot, client, instruction=spec.prompt)
        period_s = 1.0 / spec.control_hz
        for _ in range(spec.max_steps):
            started_s = monotonic()
            result = controller.step(cameras.capture())
            completed += 1
            output(
                {
                    "event": "step",
                    "runtime": spec.runtime_id,
                    "step": completed,
                    "session_revision": result.session_revision,
                }
            )
            remaining_s = period_s - (monotonic() - started_s)
            if remaining_s > 0 and completed < spec.max_steps:
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
            "runtime": spec.runtime_id,
            "steps": completed,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rlinf-so101-pi05-worker")
    parser.add_argument("--spec-json", required=True)
    args = parser.parse_args(argv)
    try:
        execute(SO101Pi05Spec.from_json(args.spec_json))
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


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeExecutionError",
    "SO101Pi05Spec",
    "execute",
    "main",
]
