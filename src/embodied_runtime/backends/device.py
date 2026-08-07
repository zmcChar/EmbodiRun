"""Description of one model-execution device."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied_runtime.types import Metadata


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    backend: str
    device_id: str
    kind: str
    vendor: str
    name: str
    total_memory_bytes: int | None = None
    capabilities: frozenset[str] = frozenset()
    metadata: Metadata = field(default_factory=dict)


__all__ = ["DeviceInfo"]
