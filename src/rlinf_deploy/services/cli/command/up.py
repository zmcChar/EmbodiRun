"""Start configured services from initialized deployment state."""

from __future__ import annotations

import argparse
import posixpath
from dataclasses import replace
from typing import Any

from ...config import config_digest
from ...executor import Command
from ...service import ServiceSpec, ServiceSupervisor
from ...state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateStore,
)
from ..context import CommandContext, ExecutorPool, print_json, state_result


class UpError(RuntimeError):
    """Configured services could not be started safely."""


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "up",
        help="start services from a matching successful init",
    )
    parser.set_defaults(command_handler=run)


def run(_args: argparse.Namespace, context: CommandContext) -> int:
    with context.executors() as executors:
        state = _up(context, executors)
    print_json(state_result("up", context.state_path, state))
    return 0


def _up(context: CommandContext, executors: ExecutorPool) -> DeploymentState:
    store = StateStore(context.state_path)
    state = store.load()
    if state is None:
        raise UpError("deployment is not initialized; run `rlinf-deploy ... init`")
    if state.config_digest != config_digest(context.config):
        raise UpError("configuration changed since init; run init again")
    missing_nodes = set(context.config.nodes) - state.nodes.keys()
    if missing_nodes:
        raise UpError(
            "initialized state is missing nodes: " + ", ".join(sorted(missing_nodes))
        )

    services = dict(state.services)
    for service in context.deployment.services:
        environment = state.environments.get(service.environment_id)
        if environment is None or environment.status != "ready":
            raise UpError(
                f"environment {service.environment_id!r} is not ready; run init again"
            )
        node = state.nodes[service.node]
        executor = executors.get(service.node)
        materialized = _materialize_service(service, node, environment)
        executable = materialized.command.argv[0]
        available = executor.run(Command(("test", "-x", executable)), check=False)
        if available.exit_code != 0:
            raise UpError(
                f"service executable is missing or not executable: {executable}"
            )
        process = ServiceSupervisor(
            executor,
            run_root=posixpath.join(node.root, "run"),
            log_root=posixpath.join(node.root, "logs"),
        ).start(materialized)
        services[service.service_id] = ServiceState(
            service_id=service.service_id,
            node=service.node,
            status="running",
            pid=process.pid,
            endpoint=service.endpoint,
        )
        state = replace(state, services=dict(services))
        store.save(state)
    return state


def _materialize_service(
    service: ServiceSpec,
    node: NodeState,
    environment: EnvironmentState,
) -> ServiceSpec:
    argv = list(service.command.argv)
    argv[0] = posixpath.join(environment.path, "bin", posixpath.basename(argv[0]))
    try:
        adapter_index = argv.index("--adapter-config") + 1
    except ValueError:
        pass
    else:
        argv[adapter_index] = _configured_path(argv[adapter_index], node.deploy_project)
    return replace(
        service,
        command=Command(
            tuple(argv),
            cwd=node.inference_project,
            environment=service.command.environment,
        ),
    )


def _configured_path(path: str, project_dir: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(project_dir, path))


__all__ = ["UpError", "register", "run"]
