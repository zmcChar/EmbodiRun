"""Shared dependencies supplied to deployment subcommands."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import DeploymentConfig, NodeConfig
from ..executor import Executor
from ..service import DeploymentPlan
from ..state import DeploymentState


class ExecutorPool:
    """Create at most one executor per node and close all of them together."""

    def __init__(
        self,
        config: DeploymentConfig,
        executor_factory: Callable[[NodeConfig], Executor],
    ) -> None:
        self._config = config
        self._executor_factory = executor_factory
        self._executors: dict[str, Executor] = {}

    def get(self, node_id: str) -> Executor:
        executor = self._executors.get(node_id)
        if executor is None:
            executor = self._executor_factory(self._config.nodes[node_id])
            self._executors[node_id] = executor
        return executor

    def close(self) -> None:
        for executor in self._executors.values():
            executor.close()
        self._executors.clear()


@dataclass(frozen=True, slots=True)
class CommandContext:
    config: DeploymentConfig
    deployment: DeploymentPlan
    state_path: Path
    executor_factory: Callable[[NodeConfig], Executor]

    @contextmanager
    def executors(self) -> Iterator[ExecutorPool]:
        pool = ExecutorPool(
            self.config,
            self.executor_factory,
        )
        try:
            yield pool
        finally:
            pool.close()


def state_path(directory: Path | None, deployment_name: str) -> Path:
    root = (
        Path.home() / ".local" / "state" / "rlinf-deploy"
        if directory is None
        else directory.expanduser()
    )
    return root / f"{deployment_name}.json"


def state_result(
    command: str,
    path: Path,
    state: DeploymentState,
) -> dict[str, object]:
    return {
        "ok": True,
        "command": command,
        "deployment": state.name,
        "state": str(path),
        "nodes": [asdict(item) for _, item in sorted(state.nodes.items())],
        "environments": [
            asdict(item) for _, item in sorted(state.environments.items())
        ],
        "services": [asdict(item) for _, item in sorted(state.services.items())],
    }


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


__all__ = [
    "CommandContext",
    "ExecutorPool",
    "print_json",
    "state_path",
    "state_result",
]
