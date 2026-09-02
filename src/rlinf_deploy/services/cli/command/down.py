"""Stop configured deployment services without touching robot runtimes."""

from __future__ import annotations

import argparse
import posixpath
from dataclasses import replace
from typing import Any

from ...service import ServiceSupervisor
from ...state import ServiceState, StateStore
from ..context import CommandContext, print_json, state_result


class DownError(RuntimeError):
    """Initialized services could not be stopped safely."""


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

    services = dict(state.services)
    with context.executors() as executors:
        for service_id, service in sorted(state.services.items()):
            node = state.nodes.get(service.node)
            if node is None or service.node not in context.config.nodes:
                raise DownError(
                    f"initialized service {service_id!r} references unavailable "
                    f"node {service.node!r}"
                )
            supervisor = ServiceSupervisor(
                executors.get(service.node),
                run_root=posixpath.join(node.root, "run"),
                log_root=posixpath.join(node.root, "logs"),
            )
            supervisor.stop(service_id)
            services[service_id] = ServiceState(
                service_id=service_id,
                node=service.node,
                status="stopped",
                endpoint=service.endpoint,
            )
            state = replace(state, services=dict(services))
            store.save(state)

    print_json(state_result("down", context.state_path, state))
    return 0


__all__ = ["DownError", "register", "run"]
