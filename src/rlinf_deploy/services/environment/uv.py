"""Synchronize locked uv environments on deployment nodes."""

from __future__ import annotations

import posixpath

from ..executor import Command, CommandResult, Executor
from .profile import EnvironmentProfile


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
        return Command(
            argv=tuple(argv),
            cwd=project_dir,
            environment={"UV_PROJECT_ENVIRONMENT": profile.path},
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


__all__ = ["UvEnvironmentManager"]
