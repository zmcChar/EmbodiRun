"""Reusable model-adapter mechanics without execution or device policy.

The public contract deliberately separates one logical request from batching:

* :meth:`preprocess_one` returns a payload whose tensor leaves retain ``B=1``;
* :meth:`collate` concatenates those payloads along their existing batch axis;
* :meth:`unbatch` removes exactly that axis and returns exactly ``batch_size``
  logical outputs; and
* :meth:`postprocess_one` interprets one unbatched model result.

Concrete adapters own model-specific semantics.  This base class only provides
cardinality-preserving tree operations and temporary source compatibility for
the prototype's original ``preprocess``/``postprocess`` spelling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

from embodied_runtime.contracts import ModelPackage, ModelPackageError, ModelSpec, TensorTree

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - minimal core installation
        raise ModelPackageError(
            "the built-in model adapters require PyTorch; install the 'torch' extra"
        ) from exc
    return torch


def _same_mapping_keys(samples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    keys = tuple(samples[0])
    expected = set(keys)
    for index, sample in enumerate(samples[1:], start=1):
        if set(sample) != expected:
            raise ModelPackageError(
                f"cannot collate sample {index}: mapping keys differ from sample 0"
            )
    return keys


def _collate_tree(samples: Sequence[Any], *, path: str = "payload") -> Any:
    """Concatenate matching B=1 tensor trees along their existing batch axis."""

    first = samples[0]
    torch = _torch()
    if isinstance(first, torch.Tensor):
        for index, sample in enumerate(samples):
            if not isinstance(sample, torch.Tensor):
                raise ModelPackageError(f"cannot collate {path}: sample {index} is not a tensor")
            if sample.ndim == 0 or sample.shape[0] != 1:
                raise ModelPackageError(
                    f"preprocess_one must retain B=1 at {path}; "
                    f"sample {index} has shape {tuple(sample.shape)}"
                )
        try:
            return torch.cat(tuple(samples), dim=0)
        except RuntimeError as exc:
            raise ModelPackageError(f"cannot collate tensor leaves at {path}: {exc}") from exc

    if isinstance(first, Mapping):
        if not all(isinstance(sample, Mapping) for sample in samples):
            raise ModelPackageError(f"cannot collate {path}: tree node types differ")
        mappings = tuple(samples)
        keys = _same_mapping_keys(mappings)
        return {
            key: _collate_tree(
                tuple(sample[key] for sample in mappings),
                path=f"{path}.{key}",
            )
            for key in keys
        }

    if isinstance(first, tuple):
        if not all(isinstance(sample, tuple) and len(sample) == len(first) for sample in samples):
            raise ModelPackageError(f"cannot collate {path}: tuple structures differ")
        # Tuple slots are structural (for example one slot per camera), so zip
        # by slot and concatenate each slot's B=1 tensor rather than stacking
        # the tuple itself.
        return tuple(
            _collate_tree(
                tuple(sample[index] for sample in samples),
                path=f"{path}[{index}]",
            )
            for index in range(len(first))
        )

    if isinstance(first, list):
        if not all(isinstance(sample, list) and len(sample) == len(first) for sample in samples):
            raise ModelPackageError(f"cannot collate {path}: list structures differ")
        return [
            _collate_tree(
                tuple(sample[index] for sample in samples),
                path=f"{path}[{index}]",
            )
            for index in range(len(first))
        ]

    if not all(sample == first for sample in samples[1:]):
        raise ModelPackageError(
            f"cannot collate non-tensor values at {path}: values are request-specific"
        )
    return first


def _unbatch_tree(value: Any, index: int, batch_size: int, *, path: str = "output") -> Any:
    """Select one item from every tensor leaf and preserve tree structure."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ModelPackageError(
                f"cannot unbatch {path}: expected leading dimension {batch_size}, "
                f"got shape {tuple(value.shape)}"
            )
        return value[index]
    if isinstance(value, Mapping):
        return {
            key: _unbatch_tree(item, index, batch_size, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _unbatch_tree(item, index, batch_size, path=f"{path}[{item_index}]")
            for item_index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _unbatch_tree(item, index, batch_size, path=f"{path}[{item_index}]")
            for item_index, item in enumerate(value)
        ]
    # Static metadata can be shared by all logical outputs.
    return value


class BaseModelAdapter(ABC, Generic[RequestT, ResultT]):
    """Optional internal base class implementing the public adapter protocol.

    External adapters only need to satisfy ``contracts.ModelAdapter`` and are
    not required to inherit this class.
    """

    @abstractmethod
    def describe(self) -> ModelSpec:
        """Describe backend-independent model identity and I/O semantics."""

    @abstractmethod
    def build_package(self, checkpoint: str, **options: Any) -> ModelPackage:
        """Load model semantics and return backend-portable entrypoints."""

    @abstractmethod
    def preprocess_one(self, request: RequestT) -> TensorTree:
        """Prepare one request while retaining a leading batch dimension of one."""

    def collate(self, samples: Sequence[TensorTree]) -> TensorTree:
        """Merge B=1 payloads without changing their model-specific structure."""

        if not samples:
            raise ModelPackageError("collate requires at least one preprocessed sample")
        return _collate_tree(samples)

    def unbatch(self, outputs: TensorTree, batch_size: int) -> Sequence[TensorTree]:
        """Remove the leading batch axis and return exactly ``batch_size`` items."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        return tuple(_unbatch_tree(outputs, index, batch_size) for index in range(batch_size))

    @abstractmethod
    def postprocess_one(self, output: TensorTree) -> ResultT:
        """Interpret one output after its batch dimension has been removed."""

    # These two wrappers preserve source compatibility with the prototype's
    # original API. New integrations should call the four cardinality-explicit
    # methods above.
    def preprocess(self, requests: Sequence[RequestT]) -> TensorTree:
        return self.collate(tuple(self.preprocess_one(request) for request in requests))

    def postprocess(self, outputs: TensorTree) -> Sequence[ResultT]:
        raise NotImplementedError(
            "legacy postprocess cannot infer generic output cardinality; call "
            "unbatch(outputs, batch_size) and postprocess_one(output)"
        )


__all__ = ["BaseModelAdapter"]
