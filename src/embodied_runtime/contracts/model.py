"""Contracts owned jointly by the model and execution-engine groups."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from .plan import ExecutionPlan, IterativeFlowPlan
from .types import Metadata, TensorTree

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Backend-independent identity and I/O facts for a model family."""

    model_id: str
    family: str
    revision: str | None = None
    modalities: tuple[str, ...] = ()
    action_dim: int | None = None
    action_horizon: int | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    """One independently executable and potentially compilable model stage."""

    name: str
    description: str = ""
    batchable: bool = True
    safe_point_after: bool = False
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ModelPackage:
    """Portable model stages plus the execution plan that composes them.

    `entrypoints` contains callables for eager/reference backends. Other
    backends may export or compile them into vendor-specific artifacts.
    """

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

    @property
    def recipe(self) -> IterativeFlowPlan:
        """Read-only compatibility view for legacy flow-package consumers."""

        if not isinstance(self.plan, IterativeFlowPlan):
            raise AttributeError("only iterative-flow packages expose the legacy recipe view")
        return self.plan


@dataclass(slots=True)
class RawRequest:
    observation: Mapping[str, Any]
    prompt: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ActionChunk:
    actions: TensorTree
    request_id: str | None = None
    metadata: Metadata = field(default_factory=dict)


@runtime_checkable
class ModelAdapter(Protocol[RequestT, ResultT]):
    """Model semantic boundary, independent of batching and hardware execution."""

    def describe(self) -> ModelSpec: ...

    def build_package(self, checkpoint: str, **options: Any) -> ModelPackage: ...

    def preprocess_one(self, request: RequestT) -> TensorTree: ...

    def collate(self, samples: Sequence[TensorTree]) -> TensorTree: ...

    def unbatch(self, outputs: TensorTree, batch_size: int) -> Sequence[TensorTree]: ...

    def postprocess_one(self, output: TensorTree) -> ResultT: ...
