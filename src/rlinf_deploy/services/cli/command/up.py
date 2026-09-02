"""Start configured services from initialized deployment state."""

from __future__ import annotations

import argparse
import math
import posixpath
import time
from dataclasses import replace
from typing import Any

from ...config import config_digest
from ...executor import Command, Executor
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


DEFAULT_WAIT_TIMEOUT_S = 600.0
_HEALTH_REQUEST_TIMEOUT_S = 2.0
_READY_POLL_INTERVAL_S = 1.0
_HEALTHCHECK_SCRIPT = """\
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])) as response:
        payload = json.load(response)
        ready = response.status == 200 and isinstance(payload, dict) and payload.get("status") == "ok"
except (OSError, ValueError):
    ready = False

raise SystemExit(0 if ready else 1)
"""


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "up",
        help="start services and wait until they are healthy",
    )
    parser.add_argument(
        "--wait-timeout",
        type=_positive_seconds,
        default=DEFAULT_WAIT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"maximum service readiness wait (default: {DEFAULT_WAIT_TIMEOUT_S:g})",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    with context.executors() as executors:
        state = _up(context, executors, wait_timeout_s=args.wait_timeout)
    print_json(state_result("up", context.state_path, state))
    return 0


def _up(
    context: CommandContext,
    executors: ExecutorPool,
    *,
    wait_timeout_s: float,
) -> DeploymentState:
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
        supervisor = ServiceSupervisor(
            executor,
            run_root=posixpath.join(node.root, "run"),
            log_root=posixpath.join(node.root, "logs"),
        )
        process = supervisor.start(materialized)
        try:
            _wait_until_ready(
                materialized,
                environment,
                supervisor,
                executor,
                pid=process.pid,
                log=process.log,
                timeout_s=wait_timeout_s,
            )
        except UpError:
            services[service.service_id] = ServiceState(
                service_id=service.service_id,
                node=service.node,
                status="failed",
                pid=process.pid,
                endpoint=service.endpoint,
            )
            state = replace(state, services=dict(services))
            store.save(state)
            raise
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


def _wait_until_ready(
    service: ServiceSpec,
    environment: EnvironmentState,
    supervisor: ServiceSupervisor,
    executor: Executor,
    *,
    pid: int | None,
    log: str | None,
    timeout_s: float,
) -> None:
    if pid is None:
        raise UpError(f"service {service.service_id!r} started without a PID")
    deadline = time.monotonic() + timeout_s
    python = posixpath.join(environment.path, "bin", "python")
    while True:
        process = supervisor.status(service.service_id)
        if process.state != "running" or process.pid != pid:
            raise UpError(
                f"service {service.service_id!r} exited before becoming ready; "
                f"log: {log}"
            )
        remaining_s = deadline - time.monotonic()
        request_timeout_s = min(
            _HEALTH_REQUEST_TIMEOUT_S,
            max(0.01, remaining_s),
        )
        health = executor.run(
            Command(
                (
                    python,
                    "-c",
                    _HEALTHCHECK_SCRIPT,
                    service.health_endpoint,
                    str(request_timeout_s),
                ),
                timeout_s=request_timeout_s + 1.0,
            ),
            check=False,
        )
        if health.exit_code == 0:
            return
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise UpError(
                f"service {service.service_id!r} did not become healthy at "
                f"{service.health_endpoint} within {timeout_s:g} seconds; "
                f"log: {log}"
            )
        time.sleep(min(_READY_POLL_INTERVAL_S, remaining_s))


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


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return seconds


__all__ = ["UpError", "register", "run"]
