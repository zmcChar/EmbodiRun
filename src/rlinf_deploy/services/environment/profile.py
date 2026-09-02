"""Resolve reusable uv environment profiles from deployment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rlinf_deploy.robots import robot_definition

from ..config import DeploymentConfig
from .errors import EnvironmentError

_MODEL_PYTHON: dict[str, str] = {
    "pi05": "3.12",
}


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """One reusable environment on one deployment node."""

    environment_id: str
    node: str
    project: Literal["deploy", "inference"]
    group: str
    path: str
    python: str | None = None
    package_index: str | None = None
    packages: tuple[str, ...] = ()


def environment_profiles(config: DeploymentConfig) -> tuple[EnvironmentProfile, ...]:
    """Return the distinct uv environments needed by the deployment.

    Robot and inference environments intentionally stay separate even when they
    reside on the same physical node. Multiple instances with the same profile
    on a node share one environment.
    """

    profiles: list[EnvironmentProfile] = []
    for robot in sorted(config.robots.values(), key=lambda item: item.robot_id):
        try:
            group, python = robot_environment_profile(robot.kind)
        except EnvironmentError:
            raise EnvironmentError(
                f"robot {robot.robot_id!r} has no environment profile for type "
                f"{robot.kind!r}"
            ) from None
        profiles.append(
            EnvironmentProfile(
                environment_id=f"{robot.node}:deploy:{group}",
                node=robot.node,
                project="deploy",
                group=group,
                path=f".venv-{group}",
                python=python,
            )
        )

    for model in sorted(config.models.values(), key=lambda item: item.model_id):
        if model.backend != "vvla":
            raise EnvironmentError(
                f"model {model.model_id!r} uses unsupported backend {model.backend!r}"
            )
        group = model.kind
        path = model.environment or f".venv-vvla-{group}"
        profiles.append(
            EnvironmentProfile(
                environment_id=f"{model.node}:inference:{group}:{path}",
                node=model.node,
                project="inference",
                group=group,
                path=path,
                python=model.python or _MODEL_PYTHON.get(group),
                package_index=model.environment_index,
                packages=model.environment_packages,
            )
        )

    return _deduplicate(profiles)


def robot_environment_profile(kind: str) -> tuple[str, str | None]:
    """Resolve a robot adapter type to its Deploy dependency group."""

    try:
        definition = robot_definition(kind)
    except (KeyError, TypeError):
        raise EnvironmentError(f"unsupported robot type {kind!r}") from None
    return definition.environment_group, definition.python


def _deduplicate(profiles: list[EnvironmentProfile]) -> tuple[EnvironmentProfile, ...]:
    by_identity: dict[tuple[str, str], EnvironmentProfile] = {}
    for profile in profiles:
        identity = (profile.node, profile.path)
        previous = by_identity.get(identity)
        if previous is not None and previous != profile:
            raise EnvironmentError(
                f"environment path {profile.path!r} on node {profile.node!r} is "
                "assigned incompatible profiles"
            )
        by_identity[identity] = profile
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (item.node, item.project, item.group, item.path),
        )
    )


__all__ = [
    "EnvironmentProfile",
    "environment_profiles",
    "robot_environment_profile",
]
