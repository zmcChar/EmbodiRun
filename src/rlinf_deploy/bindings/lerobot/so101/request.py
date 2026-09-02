"""Serializable request for an SO-101 binding worker."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlinf_deploy.bindings import BindingRun, BindingRunRequest
from rlinf_deploy.robots.lerobot.so101 import SO101Config
from rlinf_deploy.robots.sensors import SensorInput
from rlinf_deploy.services.config.robot import parse_so101_config

REQUEST_SCHEMA = "rlinf.worker.lerobot-so101.v1"


class SO101WorkerRequestError(ValueError):
    """The SO-101 worker request is invalid."""


@dataclass(frozen=True, slots=True)
class SO101WorkerRequest:
    runtime_id: str
    binding_kind: str
    prompt: str
    model_endpoint: str
    robot: SO101Config
    inputs: tuple[SensorInput, ...]
    max_steps: int
    control_hz: float
    request_timeout_s: float

    def to_json(self) -> str:
        """Encode the complete worker request for remote execution."""

        calibration_dir = self.robot.calibration_dir
        return json.dumps(
            {
                "schema": REQUEST_SCHEMA,
                "runtime_id": self.runtime_id,
                "binding_kind": self.binding_kind,
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
                "inputs": [
                    {
                        "sensor_id": item.sensor_id,
                        "name": item.name,
                        "type": item.kind,
                        "options": dict(item.options),
                    }
                    for item in self.inputs
                ],
                "max_steps": self.max_steps,
                "control_hz": self.control_hz,
                "request_timeout_s": self.request_timeout_s,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> SO101WorkerRequest:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise SO101WorkerRequestError(
                f"worker request is invalid JSON: {error}"
            ) from error
        root = _mapping(payload, "worker request")
        if root.get("schema") != REQUEST_SCHEMA:
            raise SO101WorkerRequestError("unsupported worker request schema")
        inputs_value = root.get("inputs")
        if not isinstance(inputs_value, list) or not inputs_value:
            raise SO101WorkerRequestError("worker inputs must be a non-empty list")
        inputs = tuple(
            _sensor_input(item, index) for index, item in enumerate(inputs_value)
        )
        _validate_unique_inputs(inputs)
        return cls(
            runtime_id=_string(root, "runtime_id"),
            binding_kind=_string(root, "binding_kind"),
            prompt=_string(root, "prompt"),
            model_endpoint=_string(root, "model_endpoint"),
            robot=_robot_config(root),
            inputs=inputs,
            max_steps=_positive_integer(root, "max_steps"),
            control_hz=_positive_number(root, "control_hz"),
            request_timeout_s=_positive_number(root, "request_timeout_s"),
        )


def build_run(request: BindingRunRequest) -> BindingRun:
    """Build the remote invocation for one configured SO-101 runtime."""

    robot = parse_so101_config(
        request.robot_id,
        request.robot_kind,
        request.robot_options,
    )
    inputs = tuple(request.inputs)
    if not inputs:
        raise SO101WorkerRequestError(
            f"runtime {request.runtime_id!r} requires at least one sensor input"
        )
    _validate_unique_inputs(inputs)
    worker_request = SO101WorkerRequest(
        runtime_id=request.runtime_id,
        binding_kind=request.binding_kind,
        prompt=request.prompt,
        model_endpoint=request.model_endpoint,
        robot=robot,
        inputs=inputs,
        max_steps=request.max_steps,
        control_hz=request.control_hz,
        request_timeout_s=request.request_timeout_s,
    )
    return BindingRun(arguments=("--request-json", worker_request.to_json()))


def _sensor_input(value: object, index: int) -> SensorInput:
    item = _mapping(value, f"runtime inputs[{index}]")
    try:
        return SensorInput(
            sensor_id=_string(item, "sensor_id"),
            name=_string(item, "name"),
            kind=_string(item, "type"),
            options=_mapping(item.get("options"), f"runtime inputs[{index}].options"),
        )
    except ValueError as error:
        raise SO101WorkerRequestError(f"worker input is invalid: {error}") from error


def _validate_unique_inputs(inputs: tuple[SensorInput, ...]) -> None:
    names = [item.name for item in inputs]
    if len(names) != len(set(names)):
        raise SO101WorkerRequestError("worker input names must be unique")


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
        raise SO101WorkerRequestError(
            f"runtime robot configuration is invalid: {error}"
        ) from error


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SO101WorkerRequestError(f"{name} must be an object with string keys")
    return dict(value)


def _string(value: Mapping[str, object], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise SO101WorkerRequestError(f"worker {name} must be a non-empty string")
    return result


def _optional_string(value: Mapping[str, object], name: str) -> str | None:
    result = value.get(name)
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip():
        raise SO101WorkerRequestError(
            f"worker {name} must be a non-empty string when provided"
        )
    return result


def _positive_integer(value: Mapping[str, object], name: str) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise SO101WorkerRequestError(f"worker {name} must be a positive integer")
    return result


def _positive_number(value: Mapping[str, object], name: str) -> float:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise SO101WorkerRequestError(f"worker {name} must be a positive number")
    number = float(result)
    if not math.isfinite(number) or number <= 0:
        raise SO101WorkerRequestError(f"worker {name} must be a positive number")
    return number


__all__ = [
    "REQUEST_SCHEMA",
    "SO101WorkerRequest",
    "SO101WorkerRequestError",
    "build_run",
]
