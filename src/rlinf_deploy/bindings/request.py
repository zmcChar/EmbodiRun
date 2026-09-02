"""Serializable input for the generic robot-policy binding worker."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rlinf_deploy.robots.sensors import SensorInput

REQUEST_SCHEMA = "rlinf.binding-worker.v1"


class BindingWorkerRequestError(ValueError):
    """A binding worker request is invalid."""


@dataclass(frozen=True, slots=True)
class BindingWorkerRequest:
    """Everything the generic worker needs to execute one configured runtime."""

    runtime_id: str
    binding_kind: str
    prompt: str
    model_endpoint: str
    robot_id: str
    robot_kind: str
    robot_options: Mapping[str, Any]
    inputs: tuple[SensorInput, ...]
    runtime_options: Mapping[str, Any]
    max_steps: int
    control_hz: float
    request_timeout_s: float

    def __post_init__(self) -> None:
        for name in (
            "runtime_id",
            "binding_kind",
            "prompt",
            "model_endpoint",
            "robot_id",
            "robot_kind",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BindingWorkerRequestError(f"{name} must not be empty")
        robot_options = _mapping(self.robot_options, "robot options")
        runtime_options = _mapping(self.runtime_options, "runtime options")
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise BindingWorkerRequestError("inputs must be a non-empty tuple")
        if any(not isinstance(item, SensorInput) for item in self.inputs):
            raise BindingWorkerRequestError("inputs must contain SensorInput values")
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise BindingWorkerRequestError("input names must be unique")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise BindingWorkerRequestError("max_steps must be a positive integer")
        for name in ("control_hz", "request_timeout_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise BindingWorkerRequestError(
                    f"{name} must be a finite positive number"
                )
        object.__setattr__(self, "robot_options", robot_options)
        object.__setattr__(self, "runtime_options", runtime_options)

    def to_json(self) -> str:
        """Encode the complete request for execution on a deployment node."""

        return json.dumps(
            {
                "schema": REQUEST_SCHEMA,
                "runtime_id": self.runtime_id,
                "binding_kind": self.binding_kind,
                "prompt": self.prompt,
                "model_endpoint": self.model_endpoint,
                "robot": {
                    "id": self.robot_id,
                    "type": self.robot_kind,
                    "options": dict(self.robot_options),
                },
                "inputs": [
                    {
                        "sensor_id": item.sensor_id,
                        "name": item.name,
                        "type": item.kind,
                        "options": dict(item.options),
                    }
                    for item in self.inputs
                ],
                "runtime_options": dict(self.runtime_options),
                "max_steps": self.max_steps,
                "control_hz": self.control_hz,
                "request_timeout_s": self.request_timeout_s,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> BindingWorkerRequest:
        """Decode and validate one worker request."""

        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise BindingWorkerRequestError(
                f"worker request is invalid JSON: {error}"
            ) from error
        root = _mapping(payload, "worker request")
        if root.get("schema") != REQUEST_SCHEMA:
            raise BindingWorkerRequestError("unsupported worker request schema")
        robot = _mapping(root.get("robot"), "worker robot")
        inputs_value = root.get("inputs")
        if not isinstance(inputs_value, list) or not inputs_value:
            raise BindingWorkerRequestError("worker inputs must be a non-empty list")
        try:
            return cls(
                runtime_id=_string(root, "runtime_id", "worker request"),
                binding_kind=_string(root, "binding_kind", "worker request"),
                prompt=_string(root, "prompt", "worker request"),
                model_endpoint=_string(root, "model_endpoint", "worker request"),
                robot_id=_string(robot, "id", "worker robot"),
                robot_kind=_string(robot, "type", "worker robot"),
                robot_options=_mapping(
                    robot.get("options"),
                    "worker robot options",
                ),
                inputs=tuple(
                    _sensor_input(item, index)
                    for index, item in enumerate(inputs_value)
                ),
                runtime_options=_mapping(
                    root.get("runtime_options"),
                    "worker runtime options",
                ),
                max_steps=root.get("max_steps"),
                control_hz=root.get("control_hz"),
                request_timeout_s=root.get("request_timeout_s"),
            )
        except BindingWorkerRequestError:
            raise
        except (TypeError, ValueError) as error:
            raise BindingWorkerRequestError(str(error)) from error


def _sensor_input(value: object, index: int) -> SensorInput:
    item = _mapping(value, f"worker inputs[{index}]")
    try:
        return SensorInput(
            sensor_id=_string(item, "sensor_id", f"worker inputs[{index}]"),
            name=_string(item, "name", f"worker inputs[{index}]"),
            kind=_string(item, "type", f"worker inputs[{index}]"),
            options=_mapping(
                item.get("options"),
                f"worker inputs[{index}].options",
            ),
        )
    except ValueError as error:
        raise BindingWorkerRequestError(f"worker input is invalid: {error}") from error


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise BindingWorkerRequestError(
            f"{context} must be an object with string keys"
        )
    return dict(value)


def _string(value: Mapping[str, object], name: str, context: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise BindingWorkerRequestError(
            f"{context}.{name} must be a non-empty string"
        )
    return result


__all__ = [
    "BindingWorkerRequest",
    "BindingWorkerRequestError",
    "REQUEST_SCHEMA",
]
