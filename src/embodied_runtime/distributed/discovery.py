"""Runtime identity advertised by cloud, edge, and robot processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from embodied_runtime.contracts import Metadata


class NodeRole(StrEnum):
    CLOUD = "cloud"
    EDGE = "edge"
    ROBOT = "robot"


@dataclass(frozen=True, slots=True)
class RuntimeEndpoint:
    endpoint_id: str
    role: NodeRole
    address: str
    backend_names: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    metadata: Metadata = field(default_factory=dict)
