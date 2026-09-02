"""Local deployment command execution."""

from __future__ import annotations

import os
import subprocess

from .command import Command, CommandError, CommandResult


class LocalExecutor:
    """Execute commands directly without a shell."""

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        environment = os.environ.copy()
        environment.update(command.environment)
        try:
            completed = subprocess.run(
                command.argv,
                cwd=command.cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=command.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("local command timed out") from error
        except OSError as error:
            raise RuntimeError(f"local command could not start: {error}") from error
        result = CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and result.exit_code != 0:
            raise CommandError(result.exit_code, result.stderr)
        return result

    def close(self) -> None:
        return None


__all__ = ["LocalExecutor"]
