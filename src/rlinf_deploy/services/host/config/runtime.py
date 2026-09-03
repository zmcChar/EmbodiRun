"""Robot-to-model runtime binding configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .server import ServerConfig, parse_server
from .validation import ConfigError, mapping, string


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    runtime_id: str
    robot: str
    model: str
    binding: str
    inputs: dict[str, str]
    server: ServerConfig
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_runtime(runtime_id: str, value: dict[str, Any]) -> RuntimeConfig:
    context = f"runtimes.{runtime_id}"
    server = parse_server(value.get("server"), f"{context}.server")
    if server.bind not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError(
            f"{context}.server.bind must be a loopback address because Host "
            "connects through SSH"
        )
    inputs_value = mapping(value.get("inputs", {}), f"{context}.inputs")
    inputs: dict[str, str] = {}
    for input_name, sensor_id in inputs_value.items():
        if not input_name.strip():
            raise ConfigError(f"{context}.inputs keys must not be empty")
        if not isinstance(sensor_id, str) or not sensor_id.strip():
            raise ConfigError(
                f"{context}.inputs.{input_name} must reference a sensor ID"
            )
        inputs[input_name] = sensor_id
    return RuntimeConfig(
        runtime_id=runtime_id,
        robot=string(value, "robot", context),
        model=string(value, "model", context),
        binding=string(value, "binding", context),
        inputs=inputs,
        server=server,
        options={
            name: option
            for name, option in value.items()
            if name not in {"robot", "model", "binding", "inputs", "server"}
        },
    )


__all__ = ["RuntimeConfig"]
