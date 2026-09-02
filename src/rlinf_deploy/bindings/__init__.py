"""Explicit policy-output to hardware bindings."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any, Protocol

from rlinf_deploy.inference import PolicyObservation, PolicyResult
from rlinf_deploy.robots import RobotAction, RobotObservation
from rlinf_deploy.robots.sensors import SensorInput
from rlinf_deploy.robots.sensors.cameras import CameraFrame


class BindingMapper(Protocol):
    """Translate observations and results for one policy-robot binding."""

    policy_action_space: str

    def map_observation(
        self,
        observation: RobotObservation,
        *,
        session_id: str,
        request_id: str,
        step_id: int,
        instruction: str,
        frames: Sequence[CameraFrame],
    ) -> PolicyObservation:
        """Build one policy request without performing inference."""

    def map_result(self, result: PolicyResult) -> RobotAction:
        """Map one policy result without executing it."""


@dataclass(frozen=True, slots=True)
class BindingRunRequest:
    """Generic inputs available when launching one configured binding."""

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


@dataclass(frozen=True, slots=True)
class BindingRun:
    """Arguments passed to a binding's remote worker module."""

    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BindingDefinition:
    """Static compatibility and execution information for one binding."""

    kind: str
    robot_kind: str
    model_kind: str
    worker_module: str | None = None
    build_run: Callable[[BindingRunRequest], BindingRun] | None = None
    mapper_factory: Callable[[], BindingMapper] | None = None

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("binding kind must not be empty")
        if not self.robot_kind.strip():
            raise ValueError("binding robot kind must not be empty")
        if not self.model_kind.strip():
            raise ValueError("binding model kind must not be empty")
        executable_parts = (
            self.worker_module,
            self.build_run,
            self.mapper_factory,
        )
        if any(part is not None for part in executable_parts) and any(
            part is None for part in executable_parts
        ):
            raise ValueError(
                "binding worker_module, build_run, and mapper_factory must be "
                "declared together"
            )


_BINDING_KIND = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z")


@cache
def binding_definition(kind: str) -> BindingDefinition:
    """Load ``bindings.<kind>.BINDING_DEFINITION`` by package convention."""

    if not isinstance(kind, str) or _BINDING_KIND.fullmatch(kind) is None:
        raise KeyError(kind)
    module_name = f"{__name__}.{kind}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is not None and (
            error.name == module_name or module_name.startswith(f"{error.name}.")
        ):
            raise KeyError(kind) from None
        raise
    try:
        definition = module.BINDING_DEFINITION
    except AttributeError:
        raise TypeError(f"{module_name} does not declare BINDING_DEFINITION") from None
    if not isinstance(definition, BindingDefinition):
        raise TypeError(f"{module_name}.BINDING_DEFINITION is invalid")
    if definition.kind != kind:
        raise TypeError(
            f"{module_name}.BINDING_DEFINITION declares kind {definition.kind!r}"
        )
    return definition


__all__ = [
    "BindingMapper",
    "BindingDefinition",
    "BindingRun",
    "BindingRunRequest",
    "binding_definition",
]
