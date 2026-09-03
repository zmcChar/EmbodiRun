"""Resolved deployment service and runtime specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..environment import EnvironmentProfile
from ..executor import Command


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """A long-running process that can be started on a deployment node."""

    service_id: str
    kind: Literal["model"]
    node: str
    environment_id: str
    endpoint: str
    health_endpoint: str
    command: Command
    adapter_config_json: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """One robot-policy binding executed on the robot's node."""

    runtime_id: str
    node: str
    robot: str
    model: str
    binding: str
    environment_id: str
    model_endpoint: str


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    name: str
    deploy_commit: str
    inference_commit: str
    environments: tuple[EnvironmentProfile, ...]
    services: tuple[ServiceSpec, ...]
    runtimes: tuple[RuntimeSpec, ...]


__all__ = ["DeploymentPlan", "RuntimeSpec", "ServiceSpec"]
