"""Resource descriptions shared by the engine and hardware backends."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Metadata


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


@dataclass(frozen=True, slots=True)
class MemoryStats:
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    total_bytes: int | None = None
    free_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    minimum_memory_bytes: int | None = None
    required_capabilities: frozenset[str] = frozenset()
    preferred_dtype: str | None = None
