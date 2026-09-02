"""Execute a bounded SO-101/Pi0.5 binding loop on its deployment node."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlinf_deploy.bindings import BindingRun, BindingRunRequest
from rlinf_deploy.bindings.lerobot.so101.pi05 import Pi05SO101Runtime
from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter, SO101Config
from rlinf_deploy.robots.sensors.cameras import (
    CameraSource,
    V4L2CameraConfig,
    V4L2CameraSource,
)
from rlinf_deploy.services.config.robot import parse_so101_config


class RuntimeExecutionError(RuntimeError):
    """The configured runtime cannot execute safely."""


@dataclass(frozen=True, slots=True)
class SO101Pi05Spec:
    runtime_id: str
    prompt: str
    model_endpoint: str
    robot: SO101Config
    cameras: tuple[V4L2CameraConfig, ...]
    max_steps: int
    control_hz: float
    request_timeout_s: float

    def to_json(self) -> str:
        """Encode the complete runtime specification for remote execution."""

        calibration_dir = self.robot.calibration_dir
        return json.dumps(
            {
                "schema": "rlinf.runtime.so101-pi05.v1",
                "runtime_id": self.runtime_id,
                "prompt": self.prompt,
                "model_endpoint": self.model_endpoint,
                "robot_port": self.robot.port,
                "robot_id": self.robot.robot_id,
                "calibration_id": self.robot.calibration_id,
                "calibration_dir": (
                    str(calibration_dir) if calibration_dir is not None else None
                ),
                "disable_torque_on_disconnect": (
                    self.robot.disable_torque_on_disconnect
                ),
                "max_joint_step_deg": self.robot.max_joint_step_deg,
                "max_gripper_step": self.robot.max_gripper_step,
                "step_limit_mode": self.robot.step_limit_mode,
                "cameras": [
                    {
                        "name": camera.name,
                        "device": camera.device,
                        "width": camera.width,
                        "height": camera.height,
                        "fps": camera.fps,
                    }
                    for camera in self.cameras
                ],
                "max_steps": self.max_steps,
                "control_hz": self.control_hz,
                "request_timeout_s": self.request_timeout_s,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> SO101Pi05Spec:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise RuntimeExecutionError(
                f"runtime specification is invalid JSON: {error}"
            ) from error
        root = _mapping(payload, "runtime specification")
        if root.get("schema") != "rlinf.runtime.so101-pi05.v1":
            raise RuntimeExecutionError("unsupported runtime specification schema")
        robot = _robot_config(root)
        cameras_value = root.get("cameras")
        if not isinstance(cameras_value, list) or not cameras_value:
            raise RuntimeExecutionError("runtime cameras must be a non-empty list")
        cameras = tuple(
            _camera_spec(item, index) for index, item in enumerate(cameras_value)
        )
        names = [camera.name for camera in cameras]
        devices = [camera.device for camera in cameras]
        if len(names) != len(set(names)):
            raise RuntimeExecutionError("runtime camera names must be unique")
        if len(devices) != len(set(devices)):
            raise RuntimeExecutionError("runtime camera devices must be unique")
        return cls(
            runtime_id=_string(root, "runtime_id"),
            prompt=_string(root, "prompt"),
            model_endpoint=_string(root, "model_endpoint"),
            robot=robot,
            cameras=cameras,
            max_steps=_positive_integer(root, "max_steps"),
            control_hz=_positive_number(root, "control_hz"),
            request_timeout_s=_positive_number(root, "request_timeout_s"),
        )


def build_run(request: BindingRunRequest) -> BindingRun:
    """Build the remote invocation for one configured SO-101/Pi0.5 runtime."""

    robot = parse_so101_config(
        request.robot_id,
        request.robot_kind,
        request.robot_options,
    )
    spec = SO101Pi05Spec(
        runtime_id=request.runtime_id,
        prompt=request.prompt,
        model_endpoint=request.model_endpoint,
        robot=robot,
        cameras=_configured_cameras(request.robot_options, request.robot_id),
        max_steps=request.max_steps,
        control_hz=request.control_hz,
        request_timeout_s=request.request_timeout_s,
    )
    return BindingRun(arguments=("--spec-json", spec.to_json()))


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
    parser = argparse.ArgumentParser(prog="rlinf-so101-pi05-runtime")
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


def _camera_spec(value: object, index: int) -> V4L2CameraConfig:
    item = _mapping(value, f"runtime cameras[{index}]")
    return V4L2CameraConfig(
        name=_string(item, "name"),
        device=_string(item, "device"),
        width=_positive_integer(item, "width"),
        height=_positive_integer(item, "height"),
        fps=_positive_number(item, "fps"),
    )


def _configured_cameras(
    robot_options: Mapping[str, Any],
    robot_id: str,
) -> tuple[V4L2CameraConfig, ...]:
    sensors = robot_options.get("sensors")
    if not isinstance(sensors, Mapping) or not sensors:
        raise RuntimeExecutionError(
            f"robot {robot_id!r} requires at least one camera sensor"
        )
    cameras: list[V4L2CameraConfig] = []
    for name, value in sensors.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise RuntimeExecutionError(
                f"robot {robot_id!r} sensors must be named objects"
            )
        if value.get("type") != "v4l2":
            raise RuntimeExecutionError(
                f"robot {robot_id!r} sensor {name!r} must use type 'v4l2'"
            )
        item = _mapping(value, f"robot {robot_id!r} sensor {name!r}")
        cameras.append(
            V4L2CameraConfig(
                name=name,
                device=_string(item, "device"),
                width=_positive_integer(item, "width"),
                height=_positive_integer(item, "height"),
                fps=_positive_number(item, "fps"),
            )
        )
    return tuple(cameras)


def _robot_config(value: Mapping[str, Any]) -> SO101Config:
    calibration_dir = _optional_string(value, "calibration_dir")
    try:
        return SO101Config(
            port=_string(value, "robot_port"),
            robot_id=_string(value, "robot_id"),
            calibration_id=_optional_string(value, "calibration_id"),
            calibration_dir=Path(calibration_dir) if calibration_dir else None,
            disable_torque_on_disconnect=value.get(
                "disable_torque_on_disconnect", True
            ),
            max_joint_step_deg=value.get("max_joint_step_deg", 12.0),
            max_gripper_step=value.get("max_gripper_step", 20.0),
            step_limit_mode=value.get("step_limit_mode", "reject"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeExecutionError(
            f"runtime robot configuration is invalid: {error}"
        ) from error


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeExecutionError(f"{name} must be an object with string keys")
    return dict(value)


def _string(value: Mapping[str, object], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise RuntimeExecutionError(f"runtime {name} must be a non-empty string")
    return result


def _optional_string(value: Mapping[str, object], name: str) -> str | None:
    result = value.get(name)
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip():
        raise RuntimeExecutionError(
            f"runtime {name} must be a non-empty string when provided"
        )
    return result


def _positive_integer(value: Mapping[str, object], name: str) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise RuntimeExecutionError(f"runtime {name} must be a positive integer")
    return result


def _positive_number(value: Mapping[str, object], name: str) -> float:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise RuntimeExecutionError(f"runtime {name} must be a positive number")
    number = float(result)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeExecutionError(f"runtime {name} must be a positive number")
    return number


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeExecutionError",
    "SO101Pi05Spec",
    "build_run",
    "execute",
    "main",
]
