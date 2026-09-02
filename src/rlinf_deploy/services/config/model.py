"""Inference model service configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation import (
    ConfigError,
    integer,
    mapping,
    optional_string,
    reject_unknown,
    string,
)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    bind: str
    port: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    backend: str
    transport: str
    kind: str
    node: str
    environment: str | None
    python: str | None
    server: ServerConfig
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_model(model_id: str, value: dict[str, Any]) -> ModelConfig:
    context = f"models.{model_id}"
    server_value = mapping(value.get("server"), f"{context}.server")
    reject_unknown(server_value, {"bind", "port"}, f"{context}.server")
    server = ServerConfig(
        bind=string(server_value, "bind", f"{context}.server"),
        port=integer(server_value.get("port"), f"{context}.server.port"),
    )
    if not 1 <= server.port <= 65535:
        raise ConfigError(f"{context}.server.port must be between 1 and 65535")
    return ModelConfig(
        model_id=model_id,
        backend=string(value, "backend", context),
        transport=string(value, "transport", context),
        kind=string(value, "type", context),
        node=string(value, "node", context),
        environment=optional_string(value, "environment", context),
        python=optional_string(value, "python", context),
        server=server,
        options=dict(value),
    )


__all__ = ["ModelConfig", "ServerConfig"]
