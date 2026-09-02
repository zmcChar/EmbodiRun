"""Resolve deployment configuration into an executable service plan."""

from __future__ import annotations

from rlinf_deploy.bindings import binding_definition

from ..config import DeploymentConfig, ModelConfig
from ..environment import (
    EnvironmentProfile,
    environment_profiles,
    robot_environment_profile,
)
from ..executor import Command
from .errors import ServiceError
from .spec import DeploymentPlan, RuntimeSpec, ServiceSpec


def build_plan(config: DeploymentConfig) -> DeploymentPlan:
    """Build a deterministic plan without contacting nodes or starting processes."""

    environments = environment_profiles(config)
    models = tuple(
        _model_service(config, model, environments)
        for model in sorted(config.models.values(), key=lambda item: item.model_id)
    )
    runtimes: list[RuntimeSpec] = []
    for runtime in sorted(config.runtimes.values(), key=lambda item: item.runtime_id):
        robot = config.robots[runtime.robot]
        model = config.models[runtime.model]
        _validate_binding(runtime.binding, robot.kind, model.kind)
        model_endpoint = _endpoint(config, model, consumer_node=robot.node)
        robot_environment = _find_environment(
            environments,
            node=robot.node,
            project="deploy",
            group=robot_environment_profile(robot.kind)[0],
        )
        runtimes.append(
            RuntimeSpec(
                runtime_id=runtime.runtime_id,
                node=robot.node,
                robot=runtime.robot,
                model=runtime.model,
                binding=runtime.binding,
                environment_id=robot_environment.environment_id,
                model_endpoint=model_endpoint,
            )
        )
    return DeploymentPlan(
        name=config.metadata.name,
        deploy_commit=config.metadata.deploy_commit,
        inference_commit=config.metadata.inference_commit,
        environments=environments,
        services=models,
        runtimes=tuple(runtimes),
    )


def _model_service(
    config: DeploymentConfig,
    model: ModelConfig,
    environments: tuple[EnvironmentProfile, ...],
) -> ServiceSpec:
    if model.backend != "vvla" or model.transport != "http":
        raise ServiceError(
            f"model {model.model_id!r} requires an unsupported "
            f"{model.backend}/{model.transport} service"
        )
    environment = _find_environment(
        environments,
        node=model.node,
        project="inference",
        group=model.kind,
        path=model.environment or f".venv-vvla-{model.kind}",
    )
    argv = ["vvla-http-serve", "--policy", model.kind]
    source = _option_string(model, "source", required=True)
    argv.extend(("--checkpoint", source))
    adapter_config = _option_string(model, "adapter_config")
    if adapter_config is not None:
        argv.extend(("--adapter-config", adapter_config))
    gpu = _option_string(model, "gpu")
    if gpu is not None:
        argv.extend(("--device", gpu))
    argv.extend(("--host", model.server.bind, "--port", str(model.server.port)))
    endpoint = _endpoint(config, model, consumer_node=model.node)
    return ServiceSpec(
        service_id=model.model_id,
        kind="model",
        node=model.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=f"{endpoint}/healthz",
        command=Command(tuple(argv)),
    )


def _endpoint(
    config: DeploymentConfig,
    model: ModelConfig,
    *,
    consumer_node: str,
) -> str:
    bind = model.server.bind
    if consumer_node == model.node:
        host = "127.0.0.1" if bind in {"0.0.0.0", "::"} else bind
    elif bind in {"127.0.0.1", "localhost", "::1"}:
        raise ServiceError(
            f"model {model.model_id!r} only binds to loopback but runtime is on "
            f"node {consumer_node!r}"
        )
    elif bind not in {"0.0.0.0", "::"}:
        host = bind
    else:
        connection = config.nodes[model.node].connection
        if connection.kind == "local":
            raise ServiceError(
                f"model {model.model_id!r} is on a local node with no advertised "
                f"address for runtime node {consumer_node!r}"
            )
        assert connection.host is not None
        host = connection.host
    return f"http://{_url_host(host)}:{model.server.port}"


def _url_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _find_environment(
    profiles: tuple[EnvironmentProfile, ...],
    *,
    node: str,
    project: str,
    group: str,
    path: str | None = None,
) -> EnvironmentProfile:
    matches = [
        profile
        for profile in profiles
        if profile.node == node
        and profile.project == project
        and profile.group == group
        and (path is None or profile.path == path)
    ]
    if len(matches) != 1:
        raise ServiceError(
            f"expected one {project}/{group} environment on node {node!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_binding(binding: str, robot_kind: str, model_kind: str) -> None:
    try:
        definition = binding_definition(binding)
    except (KeyError, TypeError):
        raise ServiceError(f"binding {binding!r} is not available") from None
    if definition.robot_kind != robot_kind or definition.model_kind != model_kind:
        raise ServiceError(
            f"binding {binding!r} does not match robot {robot_kind!r} and model "
            f"{model_kind!r}"
        )


def _option_string(
    model: ModelConfig,
    name: str,
    *,
    required: bool = False,
) -> str | None:
    value = model.options.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        if required:
            message = "must be a non-empty string"
        else:
            message = "must be a non-empty string when provided"
        raise ServiceError(f"models.{model.model_id}.{name} {message}")
    return value


__all__ = ["build_plan"]
