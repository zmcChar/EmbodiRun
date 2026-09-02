"""Execute deployment commands locally or over SSH."""

from .command import Command, CommandError, CommandResult
from .executor import Executor, executor_for
from .local import LocalExecutor
from .ssh import SshExecutor, render_posix

__all__ = [
    "Command",
    "CommandError",
    "CommandResult",
    "Executor",
    "LocalExecutor",
    "SshExecutor",
    "executor_for",
    "render_posix",
]
