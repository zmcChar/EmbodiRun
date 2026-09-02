"""Synchronize locked uv environments on deployment nodes."""

from __future__ import annotations

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

        return self.executor.run(self.sync_command(profile, project_dir=project_dir))


__all__ = ["UvEnvironmentManager"]
