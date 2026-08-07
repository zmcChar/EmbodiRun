"""CUDA Graph input preparation and stable cache keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TypeAlias


class CudaGraphInputError(TypeError):
    """An input tree cannot be represented by stable CUDA Graph buffers."""


@dataclass(frozen=True, slots=True)
class DynamicScalar:
    """A Python scalar copied into a graph-owned device buffer on replay."""

    value: bool | int | float
    dtype: Any
    device: Any


GraphSignature: TypeAlias = tuple[Any, ...]
GraphKey: TypeAlias = tuple[str, GraphSignature]


def mapping_like(
    original: Mapping[Any, Any],
    items: list[tuple[Any, Any]],
) -> Mapping[Any, Any]:
    if type(original) is dict:
        return dict(items)
    try:
        return type(original)(items)
    except (TypeError, ValueError) as error:
        raise CudaGraphInputError(
            f"mapping type {type(original).__qualname__} cannot be reconstructed"
        ) from error


def prepare_graph_inputs(
    torch: Any,
    device: Any,
    entrypoint: str,
    inputs: Any,
    dynamic_scalar_names: frozenset[str],
) -> Any:
    if not dynamic_scalar_names:
        return inputs
    if not isinstance(inputs, Mapping):
        raise CudaGraphInputError(
            f"dynamic scalar inputs for {entrypoint!r} require a top-level mapping"
        )

    missing = dynamic_scalar_names.difference(inputs)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise CudaGraphInputError(
            f"dynamic scalar input(s) are missing for {entrypoint!r}: {formatted}"
        )

    prepared: list[tuple[Any, Any]] = []
    for key, value in inputs.items():
        if key not in dynamic_scalar_names or isinstance(value, torch.Tensor):
            prepared.append((key, value))
            continue
        if isinstance(value, bool):
            dtype = torch.bool
        elif isinstance(value, int):
            dtype = torch.int64
        elif isinstance(value, float):
            # Match Python float semantics before the model casts to its
            # working dtype and avoid an intermediate float32 rounding.
            dtype = torch.float64
        else:
            raise CudaGraphInputError(
                f"dynamic scalar input {key!r} for {entrypoint!r} must be "
                f"bool, int, float, or Tensor; got {type(value).__qualname__}"
            )
        prepared.append(
            (
                key,
                DynamicScalar(
                    value=value,
                    dtype=dtype,
                    device=device,
                ),
            )
        )
    return mapping_like(inputs, prepared)


def tree_signature(
    torch: Any,
    value: Any,
    seen_storages: set[tuple[str, int]] | None = None,
) -> GraphSignature:
    if seen_storages is None:
        seen_storages = set()
    if isinstance(value, DynamicScalar):
        return ("dynamic_scalar", str(value.dtype), str(value.device))
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            raise CudaGraphInputError(f"tensor layout {value.layout} is not supported")
        if value.device.type != "cuda":
            raise CudaGraphInputError(f"tensor is on {value.device}, expected a CUDA device")
        if value.storage_offset() != 0:
            raise CudaGraphInputError(
                "tensor views with non-zero storage offsets are not supported"
            )
        if value.numel() > 0:
            storage = (str(value.device), int(value.untyped_storage().data_ptr()))
            if storage in seen_storages:
                raise CudaGraphInputError("tensor leaves must not share storage")
            seen_storages.add(storage)
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__module__,
            type(value).__qualname__,
            tuple((key, tree_signature(torch, item, seen_storages)) for key, item in value.items()),
        )
    if isinstance(value, tuple):
        return (
            "tuple",
            type(value).__module__,
            type(value).__qualname__,
            tuple(tree_signature(torch, item, seen_storages) for item in value),
        )
    if isinstance(value, list):
        return (
            "list",
            tuple(tree_signature(torch, item, seen_storages) for item in value),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    field.name,
                    tree_signature(torch, getattr(value, field.name), seen_storages),
                )
                for field in fields(value)
            ),
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("constant", type(value).__module__, type(value).__qualname__, repr(value))
    raise CudaGraphInputError(
        f"leaf type {type(value).__module__}.{type(value).__qualname__} is not supported"
    )


__all__ = ["CudaGraphInputError"]
