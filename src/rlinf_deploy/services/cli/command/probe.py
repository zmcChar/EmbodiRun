"""Probe configured nodes without changing them."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from typing import Any

from ...config import NodeConfig
from ...environment import probe_node
from ..context import CommandContext, ExecutorPool, print_json


@dataclass(frozen=True, slots=True)
class NodeProbeResult:
    """Connectivity and base-tool status for one configured node."""

    node_id: str
    node_type: str
    connection: str
    address: str
    reachable: bool
    latency_ms: float
    init_ready: bool = False
    platform: str | None = None
    machine: str | None = None
    python: str | None = None
    python_version: str | None = None
    git: str | None = None
    uv: str | None = None
    error: str | None = None


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "probe",
        help="check connectivity and base tools on every configured node",
    )
    parser.set_defaults(command_handler=run)


def run(_args: argparse.Namespace, context: CommandContext) -> int:
    with context.executors() as executors:
        probes = _probe_nodes(context, executors)
    reachable = all(probe.reachable for probe in probes)
    print_json(
        {
            "ok": reachable,
            "command": "probe",
            "deployment": context.deployment.name,
            "nodes": [asdict(probe) for probe in probes],
        }
    )
    return 0 if reachable else 1


def _probe_nodes(
    context: CommandContext,
    executors: ExecutorPool,
) -> tuple[NodeProbeResult, ...]:
    results: list[NodeProbeResult] = []
    for node_id in sorted(context.config.nodes):
        node = context.config.nodes[node_id]
        started = time.monotonic()
        try:
            probe = probe_node(executors.get(node_id))
        except (OSError, RuntimeError, ValueError) as error:
            results.append(
                NodeProbeResult(
                    node_id=node_id,
                    node_type=node.kind,
                    connection=node.connection.kind,
                    address=_node_address(node),
                    reachable=False,
                    latency_ms=_elapsed_ms(started),
                    error=str(error),
                )
            )
            continue
        init_ready = all((probe.python, probe.python_version, probe.git, probe.uv))
        results.append(
            NodeProbeResult(
                node_id=node_id,
                node_type=node.kind,
                connection=node.connection.kind,
                address=_node_address(node),
                reachable=True,
                latency_ms=_elapsed_ms(started),
                init_ready=init_ready,
                platform=probe.platform,
                machine=probe.machine,
                python=probe.python,
                python_version=probe.python_version,
                git=probe.git,
                uv=probe.uv,
            )
        )
    return tuple(results)


def _node_address(node: NodeConfig) -> str:
    connection = node.connection
    if connection.kind == "local":
        return "local"
    return f"{connection.host}:{connection.port}"


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 3)


__all__ = ["NodeProbeResult", "register", "run"]
