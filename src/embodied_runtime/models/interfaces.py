"""Structural interface implemented by model-family adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from embodied_runtime.types import TensorTree

from .package import ModelPackage
from .spec import ModelSpec

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@runtime_checkable
class ModelAdapter(Protocol[RequestT, ResultT]):
    def describe(self) -> ModelSpec: ...

    def build_package(self, checkpoint: str, **options: Any) -> ModelPackage: ...

    def preprocess_one(self, request: RequestT) -> TensorTree: ...

    def collate(self, samples: Sequence[TensorTree]) -> TensorTree: ...

    def unbatch(self, outputs: TensorTree, batch_size: int) -> Sequence[TensorTree]: ...

    def postprocess_one(self, output: TensorTree) -> ResultT: ...


__all__ = ["ModelAdapter"]
