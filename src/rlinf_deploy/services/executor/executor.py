"""Executor interface and default implementation selection."""

from __future__ import annotations

from typing import Protocol

from ..config import NodeConfig
from .command import Command, CommandResult
from .local import LocalExecutor
from .ssh import SshExecutor


class Executor(Protocol):
    def run(self, command: Command, *, check: bool = True) -> CommandResult: ...

    def close(self) -> None: ...


def executor_for(node: NodeConfig) -> Executor:
    if node.connection.kind == "local":
        return LocalExecutor()
    return SshExecutor(node.connection)


__all__ = ["Executor", "executor_for"]
