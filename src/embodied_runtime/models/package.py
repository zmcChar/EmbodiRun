"""Portable model entrypoints and their execution plan."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from uuid import uuid4

from embodied_runtime.types import Metadata, TensorTree

from .plans import ExecutionPlan
from .spec import EntrypointSpec, ModelSpec


@dataclass(slots=True)
class ModelPackage:
    spec: ModelSpec
    entrypoints: Mapping[str, Callable[..., TensorTree]]
    plan: ExecutionPlan
    checkpoint: str | None = None
    entrypoint_specs: Mapping[str, EntrypointSpec] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)
    package_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.package_id:
            raise ValueError("model package_id must not be empty")
        if not isinstance(self.plan, ExecutionPlan):
            raise TypeError("model package plan does not implement ExecutionPlan")
        missing = set(self.plan.required_entrypoints()).difference(self.entrypoints)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"model package is missing plan entrypoints: {names}")


__all__ = ["ModelPackage"]
