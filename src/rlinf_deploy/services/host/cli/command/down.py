"""Stop configured deployment services without touching robot runtimes."""

from __future__ import annotations

import argparse
import posixpath
from dataclasses import dataclass, replace
from typing import Any

from ...state import NodeState, ServiceState, StateStore
from ...supervisor import ServiceSupervisor
from ..context import CommandContext
from ..parallel import run_on_nodes


class DownError(RuntimeError):
    """Initialized services could not be stopped safely."""


@dataclass(frozen=True, slots=True)
class NodeDownResult:
    """Stopped service states produced by one node worker."""

    services: dict[str, ServiceState]
    error: Exception | None = None


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "down",
        help="stop initialized model services",
    )
    parser.set_defaults(command_handler=run)


def run(_args: argparse.Namespace, context: CommandContext) -> int:
    store = StateStore(context.state_path)
    state = store.load()
    if state is None:
        raise DownError("deployment is not initialized")

    for service_id, service in state.services.items():
        if service.node not in state.nodes or service.node not in context.config.nodes:
            raise DownError(
                f"initialized service {service_id!r} references unavailable "
                f"node {service.node!r}"
            )
    services_by_node = {
        node_id: tuple(
            service for service in state.services.values() if service.node == node_id
        )
        for node_id in state.nodes
    }
    progress = context.progress
    progress.begin("down", context.deployment.name)
    for node_id, node_services in sorted(services_by_node.items()):
        progress.add_node(node_id, total=max(1, len(node_services)))

    def stop_node(node_id: str) -> NodeDownResult:
        try:
            return _down_node(
                context,
                state.nodes[node_id],
                node_id,
                services_by_node[node_id],
            )
        except Exception as error:
            progress.fail(node_id, error)
            raise

    try:
        results = run_on_nodes(services_by_node, stop_node)
        services = dict(state.services)
        failures: dict[str, Exception] = dict(results.errors)
        for node_id, result in results.values.items():
            services.update(result.services)
            if result.error is not None:
                failures[node_id] = result.error
        state = replace(state, services=services)
        store.save(state)
        if failures:
            raise DownError(_node_error_summary("service shutdown", failures))
    except BaseException:
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    stopped = sum(service.status == "stopped" for service in state.services.values())
    progress.message(f"{stopped}/{len(state.services)} services stopped")
    return 0


def _down_node(
    context: CommandContext,
    node: NodeState,
    node_id: str,
    services: tuple[ServiceState, ...],
) -> NodeDownResult:
    progress = context.progress
    if not services:
        progress.update(node_id, "No services configured")
        progress.advance(node_id)
        progress.succeed(node_id, detail="No services")
        return NodeDownResult({})

    stopped: dict[str, ServiceState] = {}
    with context.executor(node_id) as executor:
        supervisor = ServiceSupervisor(
            executor,
            python=node.python,
            agent_path=posixpath.join(
                node.deploy_project,
                "src/rlinf_deploy/services/host/supervisor.py",
            ),
            run_root=posixpath.join(node.root, "run"),
            log_root=posixpath.join(node.root, "logs"),
        )
        for service in sorted(services, key=lambda item: item.service_id):
            try:
                progress.update(node_id, "Stopping service", detail=service.service_id)
                supervisor.stop(service.service_id)
                progress.advance(node_id)
                stopped[service.service_id] = ServiceState(
                    service_id=service.service_id,
                    node=service.node,
                    status="stopped",
                    endpoint=service.endpoint,
                )
            except (OSError, RuntimeError, ValueError) as error:
                progress.fail(node_id, error)
                return NodeDownResult(stopped, error)
    progress.succeed(node_id, detail=f"{len(services)} service(s) stopped")
    return NodeDownResult(stopped)


def _node_error_summary(command: str, errors: dict[str, Exception]) -> str:
    details = "; ".join(
        f"{node_id}: {error}" for node_id, error in sorted(errors.items())
    )
    return f"{command} failed on {len(errors)} node(s): {details}"


__all__ = ["DownError", "register", "run"]
