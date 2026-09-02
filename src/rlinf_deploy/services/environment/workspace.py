"""Resolve managed node workspaces and prepare pinned project checkouts."""

from __future__ import annotations

import posixpath
from pathlib import PurePosixPath

from ..executor import Command, Executor
from .errors import EnvironmentError


class ProjectManager:
    """Prepare a managed Git checkout at an exact configured revision."""

    def __init__(self, executor: Executor, *, git_executable: str = "git") -> None:
        self.executor = executor
        self.git_executable = git_executable

    def prepare(self, *, repository: str, revision: str, project_dir: str) -> None:
        if not repository or not revision or not project_dir:
            raise EnvironmentError("repository, revision, and project_dir are required")
        parent = posixpath.dirname(project_dir)
        self.executor.run(Command(("mkdir", "-p", parent)))
        checkout = self.executor.run(
            Command(("test", "-d", posixpath.join(project_dir, ".git"))),
            check=False,
        )
        if checkout.exit_code != 0:
            self.executor.run(
                Command(
                    (
                        self.git_executable,
                        "clone",
                        "--no-checkout",
                        repository,
                        project_dir,
                    ),
                    timeout_s=600.0,
                )
            )
        origin = self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "remote",
                    "get-url",
                    "origin",
                )
            )
        ).stdout.strip()
        if origin != repository:
            raise EnvironmentError(
                f"managed checkout {project_dir!r} has unexpected origin {origin!r}"
            )
        object_name = f"{revision}^{{commit}}"
        present = self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "cat-file",
                    "-e",
                    object_name,
                )
            ),
            check=False,
        )
        if present.exit_code != 0:
            self.executor.run(
                Command(
                    (
                        self.git_executable,
                        "-C",
                        project_dir,
                        "fetch",
                        "origin",
                        revision,
                    ),
                    timeout_s=600.0,
                )
            )
        self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "checkout",
                    "--detach",
                    revision,
                ),
                timeout_s=600.0,
            )
        )
        resolved = self.executor.run(
            Command(
                (
                    self.git_executable,
                    "-C",
                    project_dir,
                    "rev-parse",
                    "HEAD",
                    object_name,
                )
            )
        ).stdout.splitlines()
        if len(resolved) != 2 or resolved[0] != resolved[1]:
            raise EnvironmentError(
                f"managed checkout {project_dir!r} did not resolve to {revision!r}"
            )


def managed_root(home: str, base: str, deployment_name: str) -> str:
    """Resolve the deployment root against the probed node home directory."""

    if not home.startswith("/") or not base or "\x00" in base:
        raise EnvironmentError(
            "node home must be absolute and managed root must be valid"
        )
    if base == "~":
        resolved = home
    elif base.startswith("~/"):
        resolved = posixpath.join(home, base[2:])
    elif base.startswith("/"):
        resolved = base
    else:
        if ".." in PurePosixPath(base).parts:
            raise EnvironmentError("relative managed root cannot contain '..'")
        resolved = posixpath.join(home, base)
    root = posixpath.normpath(posixpath.join(resolved, deployment_name))
    if root == "/":
        raise EnvironmentError("managed deployment root cannot be filesystem root")
    return root


__all__ = ["ProjectManager", "managed_root"]
