"""Load a deployment YAML and validate relationships between its resources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .meta import MetadataConfig, parse_metadata
from .model import ModelConfig, parse_model
from .node import NodeConfig, parse_node
from .robot import RobotConfig, parse_robot
from .runtime import RuntimeConfig, parse_runtime
from .sensor import SensorConfig, parse_sensor
from .validation import ConfigError, mapping, named_section

_TOP_LEVEL_KEYS = {"metadata", "nodes", "robots", "sensors", "models", "runtimes"}


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    path: Path
    metadata: MetadataConfig
    nodes: dict[str, NodeConfig]
    robots: dict[str, RobotConfig]
    sensors: dict[str, SensorConfig]
    models: dict[str, ModelConfig]
    runtimes: dict[str, RuntimeConfig]


def load_config(path: str | Path) -> DeploymentConfig:
    """Load one deployment file without resolving secrets or touching nodes."""

    source = Path(path).expanduser().resolve()
    root = mapping(_load_yaml(source), "config")
    unknown = sorted(set(root) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"config contains unknown fields: {', '.join(unknown)}")

    metadata = parse_metadata(mapping(root.get("metadata"), "metadata"))
    nodes = named_section(root.get("nodes"), "nodes", parse_node)
    robots = named_section(root.get("robots", {}), "robots", parse_robot)
    sensors = named_section(root.get("sensors", {}), "sensors", parse_sensor)
    models = named_section(root.get("models", {}), "models", parse_model)
    runtimes = named_section(root.get("runtimes", {}), "runtimes", parse_runtime)
    if not nodes:
        raise ConfigError("nodes must contain at least one node")

    _validate_references(nodes, robots, sensors, models, runtimes)
    _validate_unique_robot_ports(robots)
    _validate_unique_service_ports(models, runtimes, robots)
    return DeploymentConfig(
        path=source,
        metadata=metadata,
        nodes=nodes,
        robots=robots,
        sensors=sensors,
        models=models,
        runtimes=runtimes,
    )


def config_digest(config: DeploymentConfig) -> str:
    """Hash the exact configuration bytes used to initialize a deployment."""

    try:
        payload = config.path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read config {config.path}: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on installed group
        raise RuntimeError(
            "YAML configuration requires the host environment; run "
            "`uv sync --frozen --no-dev --group host`"
        ) from error

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise ConfigError("YAML mapping keys must be scalar values") from error
            if duplicate:
                raise ConfigError(f"duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error


def _validate_references(
    nodes: dict[str, NodeConfig],
    robots: dict[str, RobotConfig],
    sensors: dict[str, SensorConfig],
    models: dict[str, ModelConfig],
    runtimes: dict[str, RuntimeConfig],
) -> None:
    for robot in robots.values():
        if robot.node not in nodes:
            raise ConfigError(
                f"robot {robot.robot_id!r} references unknown node {robot.node!r}"
            )
    for model in models.values():
        if model.node not in nodes:
            raise ConfigError(
                f"model {model.model_id!r} references unknown node {model.node!r}"
            )
    for sensor in sensors.values():
        if sensor.node not in nodes:
            raise ConfigError(
                f"sensor {sensor.sensor_id!r} references unknown node "
                f"{sensor.node!r}"
            )
    for runtime in runtimes.values():
        if runtime.robot not in robots:
            raise ConfigError(
                f"runtime {runtime.runtime_id!r} references unknown robot "
                f"{runtime.robot!r}"
            )
        if runtime.model not in models:
            raise ConfigError(
                f"runtime {runtime.runtime_id!r} references unknown model "
                f"{runtime.model!r}"
            )
        for input_name, sensor_id in runtime.inputs.items():
            if sensor_id not in sensors:
                raise ConfigError(
                    f"runtime {runtime.runtime_id!r} input {input_name!r} "
                    f"references unknown sensor {sensor_id!r}"
                )


def _validate_unique_robot_ports(robots: dict[str, RobotConfig]) -> None:
    owners: dict[tuple[str, str], str] = {}
    for robot in robots.values():
        if robot.port is None:
            continue
        key = (robot.node, robot.port)
        owner = owners.get(key)
        if owner is not None:
            raise ConfigError(
                f"robots {owner!r} and {robot.robot_id!r} share port {robot.port!r} "
                f"on node {robot.node!r}"
            )
        owners[key] = robot.robot_id


def _validate_unique_service_ports(
    models: dict[str, ModelConfig],
    runtimes: dict[str, RuntimeConfig],
    robots: dict[str, RobotConfig],
) -> None:
    owners: dict[tuple[str, int], str] = {}
    for model in models.values():
        key = (model.node, model.server.port)
        owner = owners.get(key)
        if owner is not None:
            raise ConfigError(
                f"models {owner!r} and {model.model_id!r} share port "
                f"{model.server.port} on node {model.node!r}"
            )
        owners[key] = model.model_id
    for runtime in runtimes.values():
        node = robots[runtime.robot].node
        key = (node, runtime.server.port)
        owner = owners.get(key)
        if owner is not None:
            raise ConfigError(
                f"services {owner!r} and {runtime.runtime_id!r} share port "
                f"{runtime.server.port} on node {node!r}"
            )
        owners[key] = runtime.runtime_id


__all__ = ["DeploymentConfig", "config_digest", "load_config"]
