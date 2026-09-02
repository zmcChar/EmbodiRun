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

from rlinf_deploy.bindings.lerobot.so101.pi05 import Pi05SO101Runtime
from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots.cameras import (
    CameraSource,
    V4L2CameraConfig,
    V4L2CameraSource,
)
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter, SO101Config


class RuntimeExecutionError(RuntimeError):
    """The configured runtime cannot execute safely."""


@dataclass(frozen=True, slots=True)
class SO101Pi05Spec:
    runtime_id: str
    prompt: str
    model_endpoint: str
    robot_port: str
    robot_id: str
    calibration_id: str | None
    calibration_dir: str | None
    disable_torque_on_disconnect: bool
    max_joint_step_deg: float
    max_gripper_step: float
    step_limit_mode: str
    cameras: tuple[V4L2CameraConfig, ...]
    max_steps: int
    control_hz: float
    request_timeout_s: float

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
            robot_port=_string(root, "robot_port"),
            robot_id=_string(root, "robot_id"),
            calibration_id=_optional_string(root, "calibration_id"),
            calibration_dir=_optional_string(root, "calibration_dir"),
            disable_torque_on_disconnect=_boolean(root, "disable_torque_on_disconnect"),
            max_joint_step_deg=_positive_number(root, "max_joint_step_deg"),
            max_gripper_step=_positive_number(root, "max_gripper_step"),
            step_limit_mode=_choice(
                root,
                "step_limit_mode",
                {"reject", "clip"},
            ),
            cameras=cameras,
            max_steps=_positive_integer(root, "max_steps"),
            control_hz=_positive_number(root, "control_hz"),
            request_timeout_s=_positive_number(root, "request_timeout_s"),
        )


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
        robot = robot_factory(
            SO101Config(
                port=spec.robot_port,
                robot_id=spec.robot_id,
                calibration_id=spec.calibration_id,
                calibration_dir=(
                    Path(spec.calibration_dir)
                    if spec.calibration_dir is not None
                    else None
                ),
                disable_torque_on_disconnect=spec.disable_torque_on_disconnect,
                max_joint_step_deg=spec.max_joint_step_deg,
                max_gripper_step=spec.max_gripper_step,
                step_limit_mode=spec.step_limit_mode,
            )
        )
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


def _boolean(value: Mapping[str, object], name: str) -> bool:
    result = value.get(name)
    if not isinstance(result, bool):
        raise RuntimeExecutionError(f"runtime {name} must be a boolean")
    return result


def _choice(
    value: Mapping[str, object],
    name: str,
    choices: set[str],
) -> str:
    result = _string(value, name)
    if result not in choices:
        raise RuntimeExecutionError(
            f"runtime {name} must be one of {sorted(choices)!r}"
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
    "execute",
    "main",
]
