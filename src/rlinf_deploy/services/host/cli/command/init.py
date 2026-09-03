"""Initialize configured nodes and their locked environments."""

from __future__ import annotations

import argparse
import posixpath
from dataclasses import dataclass
from typing import Any

from ...config import config_digest
from ...environment import (
    EnvironmentProfile,
    UvEnvironmentManager,
)
from ...executor import Command, Executor
from ...plan import DeploymentPlan
from ...probe import probe_node
from ...source import ProjectManager, active_deploy_project, managed_root
from ...state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateStore,
)
from ..context import CommandContext
from ..parallel import run_on_nodes

DEPLOY_REPOSITORY = "git@github.com:BUAA-CI-LAB/RLinf-deploy.git"
INFERENCE_REPOSITORY = "git@github.com:BUAA-CI-LAB/RLinf-inference.git"
DEFAULT_MANAGED_ROOT = ".local/share/rlinf-deploy"


class InitError(RuntimeError):
    """A deployment node could not be initialized safely."""


@dataclass(frozen=True, slots=True)
class NodeInitialization:
    """Initialized state produced independently by one node worker."""

    node: NodeState
    environments: tuple[EnvironmentState, ...]


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
    progress = context.progress
    progress.begin("init", context.deployment.name)
    for node_id in sorted(context.config.nodes):
        progress.add_node(
            node_id,
            total=3 + len(_profiles_on(context.deployment, node_id)),
        )
    try:
        state = _initialize(context, managed_root_base=args.root)
    except BaseException:
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    progress.message(
        f"State saved to {context.state_path} "
        f"({len(state.environments)} environments ready)"
    )
    return 0


def _initialize(
    context: CommandContext,
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

    def initialize(node_id: str) -> NodeInitialization:
        try:
            return _initialize_node(
                context,
                node_id,
                managed_root_base=managed_root_base,
            )
        except Exception as error:
            context.progress.fail(node_id, error)
            raise

    results = run_on_nodes(context.config.nodes, initialize)
    nodes = {node_id: result.node for node_id, result in results.values.items()}
    environments = {
        environment.environment_id: environment
        for result in results.values.values()
        for environment in result.environments
    }
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
    if results.errors:
        raise InitError(_node_error_summary("initialization", results.errors))
    return state


def _initialize_node(
    context: CommandContext,
    node_id: str,
    *,
    managed_root_base: str,
) -> NodeInitialization:
    progress = context.progress
    with context.executor(node_id) as executor:
        progress.update(node_id, "Probing node and required tools")
        probe = probe_node(executor, require_init_tools=True)
        assert probe.python is not None
        assert probe.python_version is not None
        assert probe.git is not None
        assert probe.uv is not None
        progress.advance(node_id)

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
        progress.update(node_id, "Preparing locked sources")
        _prepare_projects(context, executor, node_state, git=probe.git)
        executor.replace_symlink(
            active_deploy_project(node_state.root),
            node_state.deploy_project,
        )
        progress.advance(node_id)

        environments: list[EnvironmentState] = []
        for profile in _profiles_on(context.deployment, node_id):
            progress.update(
                node_id,
                "Synchronizing environment",
                detail=f"{profile.project}/{profile.group}",
            )
            project_dir = _project_dir(node_state, profile.project)
            UvEnvironmentManager(executor, uv_executable=probe.uv).prepare(
                profile,
                project_dir=project_dir,
            )
            environments.append(
                EnvironmentState(
                    environment_id=profile.environment_id,
                    node=node_id,
                    project=profile.project,
                    group=profile.group,
                    path=_environment_path(project_dir, profile.path),
                    status="ready",
                )
            )
            progress.advance(node_id)

        progress.update(node_id, "Verifying configured resources")
        _probe_resources(context, executor, node_state)
        progress.advance(node_id)
        progress.succeed(node_id)
        return NodeInitialization(node_state, tuple(environments))


def _profiles_on(
    deployment: DeploymentPlan,
    node_id: str,
) -> tuple[EnvironmentProfile, ...]:
    return tuple(
        profile for profile in deployment.environments if profile.node == node_id
    )


def _prepare_projects(
    context: CommandContext,
    executor: Executor,
    node: NodeState,
    *,
    git: str,
) -> None:
    projects = {
        profile.project for profile in _profiles_on(context.deployment, node.node_id)
    }
    projects.add("deploy")
    manager = ProjectManager(executor, git_executable=git)
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
    executor: Executor,
    node: NodeState,
) -> None:
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


def _node_error_summary(command: str, errors: dict[str, Exception]) -> str:
    details = "; ".join(
        f"{node_id}: {error}" for node_id, error in sorted(errors.items())
    )
    return f"{command} failed on {len(errors)} node(s): {details}"


__all__ = [
    "DEFAULT_MANAGED_ROOT",
    "DEPLOY_REPOSITORY",
    "INFERENCE_REPOSITORY",
    "InitError",
    "register",
    "run",
]
