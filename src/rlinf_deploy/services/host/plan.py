"""Resolve deployment configuration into an executable service plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from rlinf_deploy.bindings import BindingDefinition, binding_definition

from .config import DeploymentConfig, ModelConfig
from .environment import (
    EnvironmentProfile,
    environment_profiles,
    robot_environment_profile,
)
from .executor import Command


class ServiceError(ValueError):
    """A deployment configuration cannot be resolved into a valid plan."""


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """A long-running process that can be started on a deployment node."""

    service_id: str
    kind: Literal["model", "control", "sensor"]
    node: str
    environment_id: str
    endpoint: str
    health_endpoint: str
    command: Command
    adapter_config_json: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """One robot-policy binding executed on the robot's node."""

    runtime_id: str
    node: str
    robot: str
    model: str
    binding: str
    environment_id: str
    model_endpoint: str


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    """Deterministic host-side plan derived from one deployment config."""

    name: str
    deploy_commit: str
    inference_commit: str
    environments: tuple[EnvironmentProfile, ...]
    services: tuple[ServiceSpec, ...]
    runtimes: tuple[RuntimeSpec, ...]


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
        _validate_sensor_nodes(config, runtime.runtime_id, runtime.inputs, robot.node)
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
    adapter_config_json = _binding_adapter_config(config, model)
    configured_adapter = _option_string(model, "adapter_config")
    if adapter_config_json is not None and configured_adapter is not None:
        raise ServiceError(
            f"models.{model.model_id}.adapter_config conflicts with adapter "
            "configuration owned by its binding"
        )
    if configured_adapter is not None:
        argv.extend(("--adapter-config", configured_adapter))
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
        adapter_config_json=adapter_config_json,
    )


def _binding_adapter_config(
    config: DeploymentConfig,
    model: ModelConfig,
) -> str | None:
    definitions = {}
    generated: set[str | None] = set()
    for runtime in config.runtimes.values():
        if runtime.model != model.model_id:
            continue
        definition = _load_binding(runtime.binding)
        definitions[runtime.binding] = definition
        if definition.adapter_config is None:
            generated.add(None)
            continue
        expected_images = definition.adapter_config.get("image_fields")
        if expected_images is not None and set(runtime.inputs) != set(expected_images):
            raise ServiceError(
                f"runtime {runtime.runtime_id!r} inputs do not match binding "
                f"{runtime.binding!r} image fields"
            )
        try:
            generated.add(
                json.dumps(
                    {
                        **dict(definition.adapter_config),
                        "return_steps": definition.maximum_chunk_steps,
                    },
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as error:
            raise ServiceError(
                f"binding {runtime.binding!r} adapter configuration is not JSON"
            ) from error
    if len(generated) > 1:
        names = ", ".join(sorted(definitions))
        raise ServiceError(
            f"model {model.model_id!r} is shared by bindings with incompatible "
            f"adapter configurations: {names}"
        )
    return next(iter(generated), None)


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
    definition = _load_binding(binding)
    if definition.robot_kind != robot_kind or definition.model_kind != model_kind:
        raise ServiceError(
            f"binding {binding!r} does not match robot {robot_kind!r} and model "
            f"{model_kind!r}"
        )


def _load_binding(binding: str) -> BindingDefinition:
    try:
        definition = binding_definition(binding)
    except (KeyError, TypeError):
        raise ServiceError(f"binding {binding!r} is not available") from None
    return definition


def _validate_sensor_nodes(
    config: DeploymentConfig,
    runtime_id: str,
    inputs: dict[str, str],
    runtime_node: str,
) -> None:
    for input_name, sensor_id in inputs.items():
        sensor = config.sensors[sensor_id]
        if sensor.node != runtime_node:
            raise ServiceError(
                f"runtime {runtime_id!r} input {input_name!r} uses sensor "
                f"{sensor_id!r} on node {sensor.node!r}; cross-node sensor inputs "
                "are not supported yet"
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


__all__ = [
    "DeploymentPlan",
    "RuntimeSpec",
    "ServiceError",
    "ServiceSpec",
    "build_plan",
]
