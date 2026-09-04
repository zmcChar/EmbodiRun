"""Describe and synchronize the uv environments required by a deployment."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, replace
from typing import Literal

from rlinf_deploy.robots import robot_definition

from .config import DeploymentConfig
from .executor import Command, CommandResult, Executor

_MODEL_PYTHON: dict[str, str] = {
    "pi05": "3.12",
}


class EnvironmentError(ValueError):
    """A deployment environment cannot be resolved or prepared safely."""


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """One reusable environment on one deployment node."""

    environment_id: str
    node: str
    project: Literal["deploy", "inference"]
    group: str
    path: str
    python: str | None = None
    default_index: str | None = None
    package_index: str | None = None
    packages: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()


def environment_profiles(config: DeploymentConfig) -> tuple[EnvironmentProfile, ...]:
    """Return the distinct uv environments needed by the deployment."""

    profiles: list[EnvironmentProfile] = []
    wireless_robots = {
        runtime.robot
        for runtime in config.runtimes.values()
        if config.models[runtime.model].transport == "wireless"
    }
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
                default_index=config.nodes[robot.node].python_index,
                extras=("wireless",) if robot.robot_id in wireless_robots else (),
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
                default_index=config.nodes[model.node].python_index,
                package_index=model.environment_index,
                packages=model.environment_packages,
                extras=("wireless",) if model.transport == "wireless" else (),
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
        if previous is not None and replace(previous, extras=()) != replace(
            profile, extras=()
        ):
            raise EnvironmentError(
                f"environment path {profile.path!r} on node {profile.node!r} is "
                "assigned incompatible profiles"
            )
        if previous is None:
            by_identity[identity] = profile
        else:
            by_identity[identity] = replace(
                previous,
                extras=tuple(sorted(set(previous.extras) | set(profile.extras))),
            )
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (item.node, item.project, item.group, item.path),
        )
    )


class UvEnvironmentManager:
    """Prepare an environment through an injected local or SSH executor."""

    def __init__(self, executor: Executor, *, uv_executable: str = "uv") -> None:
        self.executor = executor
        self.uv_executable = uv_executable

    def sync_command(self, profile: EnvironmentProfile, *, project_dir: str) -> Command:
        argv = [
            self.uv_executable,
            "sync",
            "--frozen",
            "--no-dev",
            "--group",
            profile.group,
        ]
        if profile.python is not None:
            argv[2:2] = ["--python", profile.python]
        for extra in profile.extras:
            argv.extend(("--extra", extra))
        environment = {"UV_PROJECT_ENVIRONMENT": profile.path}
        if profile.default_index is not None:
            environment["UV_DEFAULT_INDEX"] = profile.default_index
        return Command(
            argv=tuple(argv),
            cwd=project_dir,
            environment=environment,
            timeout_s=1800.0,
        )

    def prepare(
        self,
        profile: EnvironmentProfile,
        *,
        project_dir: str,
    ) -> CommandResult:
        """Synchronize exactly the locked capability group for one profile."""

        result = self.executor.run(
            self.sync_command(profile, project_dir=project_dir)
        )
        if not profile.packages:
            return result
        return self.executor.run(
            self.package_command(profile, project_dir=project_dir)
        )

    def package_command(
        self,
        profile: EnvironmentProfile,
        *,
        project_dir: str,
    ) -> Command:
        if not profile.packages:
            raise ValueError("environment profile has no package overlay")
        environment_path = profile.path
        if not environment_path.startswith("/"):
            environment_path = posixpath.join(project_dir, environment_path)
        argv = [
            self.uv_executable,
            "pip",
            "install",
            "--python",
            posixpath.join(environment_path, "bin", "python"),
        ]
        if profile.package_index is not None:
            argv.extend(("--index-url", profile.package_index))
        argv.extend(profile.packages)
        return Command(
            argv=tuple(argv),
            cwd=project_dir,
            timeout_s=1800.0,
        )


__all__ = [
    "EnvironmentError",
    "EnvironmentProfile",
    "UvEnvironmentManager",
    "environment_profiles",
    "robot_environment_profile",
]
