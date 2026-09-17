"""Resolve deployment configuration into an executable service plan."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Literal

from rlinf_deploy.bindings import BindingDefinition, binding_definition
from rlinf_deploy.robots.sensors import SensorInput
from rlinf_deploy.services.control.contracts import ControlServiceConfig
from rlinf_deploy.services.inference.backends.sglang import sglang_server_command
from rlinf_deploy.services.inference.backends.vvla import (
    vvla_http_server_command,
    vvla_wireless_server_command,
)
from rlinf_deploy.services.simulation.contracts import SimulationServiceConfig
from rlinf_deploy.simulators import simulator_definition

from .config import DeploymentConfig, ModelConfig, RuntimeConfig
from .environment import (
    EnvironmentProfile,
    environment_profiles,
    robot_environment_profile,
)
from .executor import Command
from .network import service_host


class ServiceError(ValueError):
    """A deployment configuration cannot be resolved into a valid plan."""


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """A long-running process that can be started on a deployment node."""

    service_id: str
    kind: Literal["model", "control", "simulation", "sensor"]
    node: str
    environment_id: str
    endpoint: str
    health_endpoint: str | None
    command: Command
    adapter_config_json: str | None = None
    control_config_json: str | None = None
    simulation_config_json: str | None = None
    wireless_config_json: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """One policy binding executed beside a robot or simulator."""

    runtime_id: str
    node: str
    target_kind: Literal["robot", "simulator"]
    target_id: str
    model: str
    binding: str
    environment_id: str
    model_endpoint: str
    service_id: str
    service_endpoint: str


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    """Deterministic host-side plan derived from one deployment config."""

    name: str
    deploy_commit: str
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
    controls: list[ServiceSpec] = []
    simulations: list[ServiceSpec] = []
    for runtime in sorted(config.runtimes.values(), key=lambda item: item.runtime_id):
        model = config.models[runtime.model]
        if runtime.robot is not None:
            robot = config.robots[runtime.robot]
            _validate_binding(runtime.binding, robot.kind, model.kind)
            _validate_sensor_nodes(
                config, runtime.runtime_id, runtime.inputs, robot.node
            )
            model_endpoint = _endpoint(config, model, consumer_node=robot.node)
            environment = _find_environment(
                environments,
                node=robot.node,
                project="deploy",
                group=robot_environment_profile(robot.kind)[0],
            )
            runtime_spec, service = _control_service(
                config,
                runtime,
                environment=environment,
                model_endpoint=model_endpoint,
            )
            controls.append(service)
        else:
            simulator = config.simulators[runtime.simulator]
            definition = simulator_definition(simulator.kind)
            _validate_binding(runtime.binding, definition.embodiment_kind, model.kind)
            if runtime.inputs:
                raise ServiceError(
                    f"simulation runtime {runtime.runtime_id!r} receives rendered "
                    "inputs from its simulator and must not reference sensors"
                )
            model_endpoint = _endpoint(config, model, consumer_node=simulator.node)
            environment = _find_environment(
                environments,
                node=simulator.node,
                project="deploy",
                group=definition.environment_group,
            )
            runtime_spec, service = _simulation_service(
                config,
                runtime,
                environment=environment,
                model_endpoint=model_endpoint,
            )
            simulations.append(service)
        runtimes.append(runtime_spec)
    services = (*models, *controls, *simulations)
    if len({service.service_id for service in services}) != len(services):
        raise ServiceError(
            "model IDs must not collide with generated runtime service IDs"
        )
    wireless = _wireless_endpoint_configs(config)
    return DeploymentPlan(
        name=config.metadata.name,
        deploy_commit=config.metadata.deploy_commit,
        environments=environments,
        services=tuple(
            replace(service, wireless_config_json=wireless.get(service.service_id))
            for service in services
        ),
        runtimes=tuple(runtimes),
    )


def _model_service(
    config: DeploymentConfig,
    model: ModelConfig,
    environments: tuple[EnvironmentProfile, ...],
) -> ServiceSpec:
    if model.backend not in {"vvla", "sglang"} or model.transport not in {
        "http",
        "wireless",
    }:
        raise ServiceError(
            f"model {model.model_id!r} requires an unsupported "
            f"{model.backend}/{model.transport} service"
        )
    if model.backend == "sglang" and model.transport != "http":
        raise ServiceError(
            f"model {model.model_id!r} uses sglang, which currently requires "
            "the http transport"
        )
    environment_group = model.kind if model.backend == "vvla" else "sglang"
    environment = _find_environment(
        environments,
        node=model.node,
        project="inference",
        group=environment_group,
        path=model.environment or f".venv-{model.backend}-{model.kind}",
    )
    source = _option_string(model, "source", required=True)
    adapter_config_json = _binding_adapter_config(config, model)
    configured_adapter = _option_string(model, "adapter_config")
    if (
        model.backend == "vvla"
        and adapter_config_json is not None
        and configured_adapter is not None
    ):
        raise ServiceError(
            f"models.{model.model_id}.adapter_config conflicts with adapter "
            "configuration owned by its binding"
        )
    gpu = _option_string(model, "gpu")
    adapter_path = configured_adapter
    command_environment: dict[str, str] = {}
    if model.backend == "sglang":
        if configured_adapter is not None:
            raise ServiceError(
                f"models.{model.model_id}.adapter_config is VVLA-specific; use "
                "pipeline_config for SGLang"
            )
        argv = sglang_server_command(
            checkpoint=source,
            bind=model.server.bind,
            port=model.server.port,
            executable=_option_string(model, "server_executable") or "sglang",
            pipeline=_option_string(model, "pipeline"),
            pipeline_config=_option_string(model, "pipeline_config"),
            extra_args=_option_strings(model, "server_args"),
        )
        command_environment = _sglang_environment(model, gpu)
    elif model.transport == "http":
        argv = vvla_http_server_command(
            policy=model.kind,
            checkpoint=source,
            bind=model.server.bind,
            port=model.server.port,
            device=gpu,
            adapter_config=adapter_path,
        )
    else:
        argv = vvla_wireless_server_command(
            policy=model.kind,
            checkpoint=source,
            comm_config=f"{model.model_id}.wireless.json",
            device=gpu,
            adapter_config=adapter_path,
        )
    if model.backend == "vvla":
        argv = (*argv, *_option_strings(model, "server_args"))
    endpoint = _endpoint(config, model, consumer_node=model.node)
    return ServiceSpec(
        service_id=model.model_id,
        kind="model",
        node=model.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=(
            f"{endpoint}/health"
            if model.backend == "sglang"
            else f"{endpoint}/healthz"
            if model.transport == "http"
            else None
        ),
        command=Command(argv, environment=command_environment),
        adapter_config_json=(adapter_config_json if model.backend == "vvla" else None),
    )


def _control_service(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
    *,
    environment: EnvironmentProfile,
    model_endpoint: str,
) -> tuple[RuntimeSpec, ServiceSpec]:
    robot = config.robots[runtime.robot]
    model = config.models[runtime.model]
    service_id = f"control-{runtime.runtime_id}"
    endpoint = f"http://{_url_host(runtime.server.bind)}:{runtime.server.port}"
    inference_options = _inference_options(config, runtime, model)
    try:
        control_config = ControlServiceConfig(
            runtime_id=runtime.runtime_id,
            binding_kind=runtime.binding,
            bind=runtime.server.bind,
            port=runtime.server.port,
            inference_backend=model.backend,
            inference_transport=model.transport,
            inference_endpoint=model_endpoint,
            inference_options=inference_options,
            robot_id=robot.robot_id,
            robot_kind=robot.kind,
            robot_options=robot.options,
            inputs=tuple(
                SensorInput(
                    sensor_id=sensor_id,
                    name=input_name,
                    kind=config.sensors[sensor_id].kind,
                    options=config.sensors[sensor_id].options,
                )
                for input_name, sensor_id in runtime.inputs.items()
            ),
            runtime_options=runtime.options,
        )
        control_config_json = control_config.to_json()
    except (TypeError, ValueError) as error:
        raise ServiceError(
            f"runtime {runtime.runtime_id!r} control configuration is invalid: {error}"
        ) from error
    runtime_spec = RuntimeSpec(
        runtime_id=runtime.runtime_id,
        node=robot.node,
        target_kind="robot",
        target_id=robot.robot_id,
        model=runtime.model,
        binding=runtime.binding,
        environment_id=environment.environment_id,
        model_endpoint=model_endpoint,
        service_id=service_id,
        service_endpoint=endpoint,
    )
    service = ServiceSpec(
        service_id=service_id,
        kind="control",
        node=robot.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=f"{endpoint}/healthz",
        command=Command(("rlinf-control-serve",)),
        control_config_json=control_config_json,
    )
    return runtime_spec, service


def _simulation_service(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
    *,
    environment: EnvironmentProfile,
    model_endpoint: str,
) -> tuple[RuntimeSpec, ServiceSpec]:
    simulator = config.simulators[runtime.simulator]
    model = config.models[runtime.model]
    service_id = f"simulation-{runtime.runtime_id}"
    endpoint = f"http://{_url_host(runtime.server.bind)}:{runtime.server.port}"
    inference_options = _inference_options(config, runtime, model)
    try:
        service_config = SimulationServiceConfig(
            runtime_id=runtime.runtime_id,
            binding_kind=runtime.binding,
            bind=runtime.server.bind,
            port=runtime.server.port,
            inference_backend=model.backend,
            inference_transport=model.transport,
            inference_endpoint=model_endpoint,
            inference_options=inference_options,
            simulator_id=simulator.simulator_id,
            simulator_kind=simulator.kind,
            simulator_options=simulator.options,
        )
    except (TypeError, ValueError) as error:
        raise ServiceError(
            f"runtime {runtime.runtime_id!r} simulation configuration is invalid: "
            f"{error}"
        ) from error
    runtime_spec = RuntimeSpec(
        runtime_id=runtime.runtime_id,
        node=simulator.node,
        target_kind="simulator",
        target_id=simulator.simulator_id,
        model=runtime.model,
        binding=runtime.binding,
        environment_id=environment.environment_id,
        model_endpoint=model_endpoint,
        service_id=service_id,
        service_endpoint=endpoint,
    )
    service = ServiceSpec(
        service_id=service_id,
        kind="simulation",
        node=simulator.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=f"{endpoint}/healthz",
        command=Command(("rlinf-simulation-serve",)),
        simulation_config_json=service_config.to_json(),
    )
    return runtime_spec, service


def _inference_options(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
    model: ModelConfig,
) -> dict[str, object]:
    options: dict[str, object] = {}
    token = _option_string(model, "token")
    if token is not None:
        options["token"] = token
    if model.transport == "wireless":
        prefix = "control" if runtime.robot is not None else "simulation"
        options.update(
            {
                "comm_config": f"{prefix}-{runtime.runtime_id}.wireless.json",
                "server_node_id": _wireless_model_peer_id(model),
            }
        )
    if model.backend != "sglang":
        return options

    definition = _load_binding(runtime.binding)
    adapter = dict(definition.adapter_config or {})
    for name in ("state_fields", "action_feature_names"):
        value = adapter.get(name)
        if value is not None:
            options[name] = list(value)

    image_fields = _runtime_image_fields(config, runtime)
    image_keys = {field: field.rsplit(".", 1)[-1] for field in image_fields}
    configured_image_keys = _configured_image_keys(
        model,
        image_fields,
        runtime_id=runtime.runtime_id,
    )
    if configured_image_keys is not None:
        image_keys.update(configured_image_keys)
    if len(image_keys.values()) != len(set(image_keys.values())):
        raise ServiceError(f"models.{model.model_id}.image_keys values must be unique")
    options["image_keys"] = image_keys

    parameters = dict(_option_mapping(model, "parameters") or {})
    parameters.setdefault("action_horizon", definition.maximum_chunk_steps)
    action_horizon = parameters["action_horizon"]
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or not 1 <= action_horizon <= definition.maximum_chunk_steps
    ):
        raise ServiceError(
            f"models.{model.model_id}.parameters.action_horizon must be an "
            f"integer between 1 and {definition.maximum_chunk_steps}"
        )
    options["parameters"] = parameters
    options["runtime"] = dict(_option_mapping(model, "runtime") or {})
    output_action_dim = model.options.get("output_action_dim")
    if output_action_dim is not None:
        if (
            isinstance(output_action_dim, bool)
            or not isinstance(output_action_dim, int)
            or output_action_dim <= 0
        ):
            raise ServiceError(
                f"models.{model.model_id}.output_action_dim must be a positive integer"
            )
        options["output_action_dim"] = output_action_dim
    return options


def _binding_adapter_config(
    config: DeploymentConfig,
    model: ModelConfig,
) -> str | None:
    policy_kwargs = _option_mapping(model, "policy_kwargs")
    if policy_kwargs is not None:
        if model.backend != "vvla":
            raise ServiceError(f"models.{model.model_id}.policy_kwargs requires VVLA")
        if any(not isinstance(key, str) or not key.strip() for key in policy_kwargs):
            raise ServiceError(
                f"models.{model.model_id}.policy_kwargs keys must be non-empty strings"
            )
    runtime_ids: list[str] = []
    generated: set[str | None] = set()
    for runtime in config.runtimes.values():
        if runtime.model != model.model_id:
            continue
        runtime_ids.append(runtime.runtime_id)
        definition = _load_binding(runtime.binding)
        if definition.adapter_config is None:
            generated.add(None)
            continue
        adapter_config = dict(definition.adapter_config)
        if policy_kwargs is not None:
            adapter_config["policy_kwargs"] = dict(policy_kwargs)
        image_fields = _runtime_image_fields(config, runtime)
        if image_fields:
            adapter_config["image_fields"] = image_fields
        if model.backend == "vvla":
            image_keys = _configured_image_keys(
                model,
                image_fields,
                runtime_id=runtime.runtime_id,
            )
            if image_keys is not None:
                resolved_image_keys = {
                    field: image_keys.get(field, field) for field in image_fields
                }
                if len(resolved_image_keys.values()) != len(
                    set(resolved_image_keys.values())
                ):
                    raise ServiceError(
                        f"models.{model.model_id}.image_keys values must be unique"
                    )
                adapter_config["image_keys"] = image_keys
        try:
            generated.add(
                json.dumps(
                    {
                        **adapter_config,
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
        names = ", ".join(sorted(runtime_ids))
        raise ServiceError(
            f"model {model.model_id!r} is shared by runtimes with incompatible "
            f"adapter configurations: {names}"
        )
    result = next(iter(generated), None)
    if policy_kwargs is not None and result is None:
        raise ServiceError(
            f"models.{model.model_id}.policy_kwargs requires a binding-generated "
            "adapter configuration; otherwise put policy_kwargs in the adapter_config JSON"
        )
    return result


def _runtime_image_fields(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
) -> tuple[str, ...]:
    if runtime.robot is not None:
        return tuple(runtime.inputs)
    simulator = config.simulators[runtime.simulator]
    return simulator_definition(simulator.kind).image_fields


def _wireless_model_peer_id(model: ModelConfig) -> str:
    return f"model.{model.model_id}"


def _runtime_node(config: DeploymentConfig, runtime: RuntimeConfig) -> str:
    if runtime.robot is not None:
        return config.robots[runtime.robot].node
    return config.simulators[runtime.simulator].node


def _wireless_endpoint_configs(config: DeploymentConfig) -> dict[str, str]:
    """Map service IDs to JSON (also valid YAML) understood by WirelessComm.

    A model knows all its runtimes; each runtime knows only its selected model.
    Peer IDs identify processes, not physical machines.
    """

    result: dict[str, str] = {}
    for model in config.models.values():
        if model.transport != "wireless":
            continue
        runtimes = sorted(
            (
                runtime
                for runtime in config.runtimes.values()
                if runtime.model == model.model_id
            ),
            key=lambda runtime: runtime.runtime_id,
        )
        remote = any(
            _runtime_node(config, runtime) != model.node for runtime in runtimes
        )
        server = {
            "node_id": _wireless_model_peer_id(model),
            "host": service_host(
                config.nodes[model.node], model.server.bind, remote=remote
            ),
            "port": model.server.port,
        }
        clients = []
        for runtime in runtimes:
            client = runtime.inference_client
            assert client is not None
            node = _runtime_node(config, runtime)
            peer = {
                "node_id": f"runtime.{runtime.runtime_id}",
                "host": service_host(
                    config.nodes[node], client.server.bind, remote=node != model.node
                ),
                "port": client.server.port,
            }
            clients.append(peer)
            prefix = "control" if runtime.robot is not None else "simulation"
            result[f"{prefix}-{runtime.runtime_id}"] = json.dumps(
                {
                    "local": {**peer, "bind_host": client.server.bind},
                    "peers": [server],
                    "comm": client.transport_options,
                },
                allow_nan=False,
                sort_keys=True,
            )
        result[model.model_id] = json.dumps(
            {
                "local": {**server, "bind_host": model.server.bind},
                "peers": clients,
                "comm": model.transport_options,
            },
            allow_nan=False,
            sort_keys=True,
        )
    return result


def _endpoint(
    config: DeploymentConfig,
    model: ModelConfig,
    *,
    consumer_node: str,
) -> str:
    if model.transport == "wireless":
        return f"wireless://{_wireless_model_peer_id(model)}"
    host = service_host(
        config.nodes[model.node], model.server.bind, remote=consumer_node != model.node
    )
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


def _option_strings(model: ModelConfig, name: str) -> tuple[str, ...]:
    value = model.options.get(name, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ServiceError(f"models.{model.model_id}.{name} must be a list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ServiceError(
            f"models.{model.model_id}.{name} must contain non-empty strings"
        )
    reserved = (
        "--host",
        "--port",
        "--model-type",
        "--pipeline",
        "--pipeline-class-name",
        "--pipeline-config-path",
    )
    if any(
        item == option or item.startswith(f"{option}=")
        for item in result
        for option in reserved
    ):
        raise ServiceError(
            f"models.{model.model_id}.{name} must not override host, port, model "
            "type, or explicit pipeline settings"
        )
    return result


def _option_mapping(
    model: ModelConfig,
    name: str,
) -> dict[str, object] | None:
    value = model.options.get(name)
    if value is None:
        return None
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ServiceError(f"models.{model.model_id}.{name} must be an object")
    return dict(value)


def _configured_image_keys(
    model: ModelConfig,
    image_fields: tuple[str, ...],
    *,
    runtime_id: str,
) -> dict[str, str] | None:
    value = _option_mapping(model, "image_keys")
    if value is None:
        return None
    if any(
        not isinstance(target, str) or not target.strip() for target in value.values()
    ):
        raise ServiceError(
            f"models.{model.model_id}.image_keys values must be non-empty strings"
        )
    unknown = sorted(set(value) - set(image_fields))
    if unknown:
        raise ServiceError(
            f"models.{model.model_id}.image_keys contains fields not produced "
            f"by runtime {runtime_id!r}: {', '.join(unknown)}"
        )
    return {
        source: target for source, target in value.items() if isinstance(target, str)
    }


def _sglang_environment(
    model: ModelConfig,
    gpu: str | None,
) -> dict[str, str]:
    if gpu is None:
        return {}
    prefix = "cuda:"
    if not gpu.startswith(prefix):
        raise ServiceError(
            f"models.{model.model_id}.gpu must use cuda:<device> for SGLang"
        )
    devices = gpu.removeprefix(prefix)
    parts = devices.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ServiceError(
            f"models.{model.model_id}.gpu must use cuda:<device> for SGLang"
        )
    return {"CUDA_VISIBLE_DEVICES": ",".join(parts)}


__all__ = [
    "DeploymentPlan",
    "RuntimeSpec",
    "ServiceError",
    "ServiceSpec",
    "build_plan",
]
