"""Initialize configured nodes and their locked environments."""

from __future__ import annotations

import argparse
import posixpath
from typing import Any

from ...config import config_digest
from ...environment import (
    EnvironmentProfile,
    ProjectManager,
    UvEnvironmentManager,
    managed_root,
    probe_node,
)
from ...executor import Command, Executor
from ...service import DeploymentPlan
from ...state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateStore,
)
from ..context import CommandContext, ExecutorPool, print_json, state_result

DEPLOY_REPOSITORY = "https://github.com/BUAA-CI-LAB/RLinf-deploy.git"
INFERENCE_REPOSITORY = "https://github.com/BUAA-CI-LAB/RLinf-inference.git"
DEFAULT_MANAGED_ROOT = ".local/share/rlinf-deploy"


class InitError(RuntimeError):
    """A deployment node could not be initialized safely."""


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "init",
        help="probe nodes and synchronize locked deployment environments",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_MANAGED_ROOT,
        help=(
            "managed directory on each node, relative to its home by default "
            f"({DEFAULT_MANAGED_ROOT})"
        ),
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    with context.executors() as executors:
        state = _initialize(context, executors, managed_root_base=args.root)
    print_json(state_result("init", context.state_path, state))
    return 0


def _initialize(
    context: CommandContext,
    executors: ExecutorPool,
    *,
    managed_root_base: str,
) -> DeploymentState:
    store = StateStore(context.state_path)
    previous = store.load()
    digest = config_digest(context.config)
    if previous is not None and previous.config_digest != digest:
        running = [
            service.service_id
            for service in previous.services.values()
            if service.status == "running"
        ]
        if running:
            raise InitError(
                "configuration changed while services are recorded as running: "
                + ", ".join(sorted(running))
            )

    nodes: dict[str, NodeState] = {}
    environments: dict[str, EnvironmentState] = {}
    for node_id in sorted(context.config.nodes):
        executor = executors.get(node_id)
        probe = probe_node(executor, require_init_tools=True)
        assert probe.python is not None
        assert probe.python_version is not None
        assert probe.git is not None
        assert probe.uv is not None
        root = managed_root(probe.home, managed_root_base, context.deployment.name)
        node_state = NodeState(
            node_id=node_id,
            home=probe.home,
            root=root,
            deploy_project=posixpath.join(root, "sources", "deploy"),
            inference_project=posixpath.join(root, "sources", "inference"),
            platform=probe.platform,
            machine=probe.machine,
            python=probe.python,
            python_version=probe.python_version,
        )
        _prepare_projects(context, executors, node_state, git=probe.git)
        _probe_resources(context, executors, node_state)
        nodes[node_id] = node_state
        for profile in _profiles_on(context.deployment, node_id):
            project_dir = _project_dir(node_state, profile.project)
            UvEnvironmentManager(executor, uv_executable=probe.uv).prepare(
                profile,
                project_dir=project_dir,
            )
            environments[profile.environment_id] = EnvironmentState(
                environment_id=profile.environment_id,
                node=node_id,
                project=profile.project,
                group=profile.group,
                path=_environment_path(project_dir, profile.path),
                status="ready",
            )

    state = DeploymentState(
        name=context.deployment.name,
        config_digest=digest,
        deploy_commit=context.deployment.deploy_commit,
        inference_commit=context.deployment.inference_commit,
        nodes=nodes,
        environments=environments,
        services=_initial_service_state(context.deployment, previous, digest),
    )
    store.save(state)
    return state


def _profiles_on(
    deployment: DeploymentPlan,
    node_id: str,
) -> tuple[EnvironmentProfile, ...]:
    return tuple(
        profile for profile in deployment.environments if profile.node == node_id
    )


def _prepare_projects(
    context: CommandContext,
    executors: ExecutorPool,
    node: NodeState,
    *,
    git: str,
) -> None:
    projects = {
        profile.project for profile in _profiles_on(context.deployment, node.node_id)
    }
    relative_adapter = any(
        model.node == node.node_id
        and isinstance(model.options.get("adapter_config"), str)
        and not str(model.options["adapter_config"]).startswith("/")
        for model in context.config.models.values()
    )
    if relative_adapter:
        projects.add("deploy")
    manager = ProjectManager(executors.get(node.node_id), git_executable=git)
    if "deploy" in projects:
        manager.prepare(
            repository=DEPLOY_REPOSITORY,
            revision=context.deployment.deploy_commit,
            project_dir=node.deploy_project,
        )
    if "inference" in projects:
        manager.prepare(
            repository=INFERENCE_REPOSITORY,
            revision=context.deployment.inference_commit,
            project_dir=node.inference_project,
        )


def _probe_resources(
    context: CommandContext,
    executors: ExecutorPool,
    node: NodeState,
) -> None:
    executor = executors.get(node.node_id)
    for robot in context.config.robots.values():
        if robot.node != node.node_id:
            continue
        if robot.port is not None:
            _require_path(
                executor,
                robot.port,
                kind="e",
                description=f"robot {robot.robot_id!r} port",
            )
        calibration = robot.options.get("calibration_dir")
        if isinstance(calibration, str):
            _require_path(
                executor,
                _configured_path(calibration, node.deploy_project),
                kind="d",
                description=f"robot {robot.robot_id!r} calibration directory",
            )
    for model in context.config.models.values():
        if model.node != node.node_id:
            continue
        source = model.options.get("source")
        if isinstance(source, str) and source.startswith(("/", "./", "../")):
            _require_path(
                executor,
                _configured_path(source, node.inference_project),
                kind="e",
                description=f"model {model.model_id!r} source",
            )
        adapter = model.options.get("adapter_config")
        if isinstance(adapter, str):
            _require_path(
                executor,
                _configured_path(adapter, node.deploy_project),
                kind="f",
                description=f"model {model.model_id!r} adapter config",
            )


def _initial_service_state(
    deployment: DeploymentPlan,
    previous: DeploymentState | None,
    digest: str,
) -> dict[str, ServiceState]:
    result: dict[str, ServiceState] = {}
    preserve = previous is not None and previous.config_digest == digest
    for service in deployment.services:
        old = previous.services.get(service.service_id) if preserve else None
        if (
            old is not None
            and old.node == service.node
            and old.endpoint == service.endpoint
        ):
            result[service.service_id] = old
        else:
            result[service.service_id] = ServiceState(
                service_id=service.service_id,
                node=service.node,
                endpoint=service.endpoint,
            )
    return result


def _project_dir(node: NodeState, project: str) -> str:
    if project == "deploy":
        return node.deploy_project
    if project == "inference":
        return node.inference_project
    raise InitError(f"unsupported project role {project!r}")


def _environment_path(project_dir: str, path: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(project_dir, path))


def _configured_path(path: str, project_dir: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(project_dir, path))


def _require_path(
    executor: Executor,
    path: str,
    *,
    kind: str,
    description: str,
) -> None:
    result = executor.run(Command(("test", f"-{kind}", path)), check=False)
    if result.exit_code != 0:
        raise InitError(f"{description} was not found at {path!r}")


__all__ = [
    "DEFAULT_MANAGED_ROOT",
    "DEPLOY_REPOSITORY",
    "INFERENCE_REPOSITORY",
    "InitError",
    "register",
    "run",
]
