"""Deployment node configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .connection import ConnectionConfig, parse_connection
from .validation import mapping, optional_string, string


@dataclass(frozen=True, slots=True)
class NodeConfig:
    node_id: str
    kind: str
    connection: ConnectionConfig
    python_index: str | None
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_node(node_id: str, value: dict[str, Any]) -> NodeConfig:
    context = f"nodes.{node_id}"
    connection = parse_connection(
        mapping(value.get("connection"), f"{context}.connection")
    )
    return NodeConfig(
        node_id=node_id,
        kind=string(value, "type", context),
        connection=connection,
        python_index=optional_string(value, "python_index", context),
        options=dict(value),
    )


__all__ = ["NodeConfig"]
