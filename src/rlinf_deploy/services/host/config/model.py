"""Inference model service configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .server import ServerConfig, parse_server
from .validation import (
    ConfigError,
    optional_string,
    string,
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    backend: str
    transport: str
    kind: str
    node: str
    environment: str | None
    python: str | None
    environment_index: str | None
    environment_packages: tuple[str, ...]
    server: ServerConfig
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_model(model_id: str, value: dict[str, Any]) -> ModelConfig:
    context = f"models.{model_id}"
    environment_packages = _environment_packages(value, context)
    environment_index = optional_string(value, "environment_index", context)
    if environment_index is not None and not environment_packages:
        raise ConfigError(
            f"{context}.environment_index requires environment_packages"
        )
    server = parse_server(value.get("server"), f"{context}.server")
    return ModelConfig(
        model_id=model_id,
        backend=string(value, "backend", context),
        transport=string(value, "transport", context),
        kind=string(value, "type", context),
        node=string(value, "node", context),
        environment=optional_string(value, "environment", context),
        python=optional_string(value, "python", context),
        environment_index=environment_index,
        environment_packages=environment_packages,
        server=server,
        options=dict(value),
    )


def _environment_packages(
    value: dict[str, Any],
    context: str,
) -> tuple[str, ...]:
    packages = value.get("environment_packages")
    if packages is None:
        return ()
    if not isinstance(packages, list) or not packages:
        raise ConfigError(
            f"{context}.environment_packages must be a non-empty list"
        )
    result: list[str] = []
    for index, package in enumerate(packages):
        if (
            not isinstance(package, str)
            or not package.strip()
            or package.startswith("-")
        ):
            raise ConfigError(
                f"{context}.environment_packages[{index}] must be a package "
                "requirement"
            )
        result.append(package)
    return tuple(result)


__all__ = ["ModelConfig"]
