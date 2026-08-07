"""Backend-independent model and entrypoint descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied_runtime.types import Metadata


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    family: str
    revision: str | None = None
    modalities: tuple[str, ...] = ()
    action_dim: int | None = None
    action_horizon: int | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    name: str
    description: str = ""
    batchable: bool = True
    safe_point_after: bool = False
    metadata: Metadata = field(default_factory=dict)


__all__ = ["EntrypointSpec", "ModelSpec"]
