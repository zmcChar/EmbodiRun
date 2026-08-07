"""Static CUDA Graph input buffers and detached output trees."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

from .cuda_graph_key import CudaGraphInputError, DynamicScalar, mapping_like


def make_static_tree(torch: Any, value: Any) -> Any:
    if isinstance(value, DynamicScalar):
        return torch.tensor(
            value.value,
            dtype=value.dtype,
            device=value.device,
        )
    if isinstance(value, torch.Tensor):
        static = torch.empty_strided(
            tuple(value.shape),
            tuple(value.stride()),
            dtype=value.dtype,
            device=value.device,
        )
        static.copy_(value)
        return static
    if isinstance(value, Mapping):
        return mapping_like(
            value,
            [(key, make_static_tree(torch, item)) for key, item in value.items()],
        )
    if isinstance(value, tuple):
        items = tuple(make_static_tree(torch, item) for item in value)
        if type(value) is tuple:
            return items
        if hasattr(value, "_fields"):
            return type(value)(*items)
        try:
            return type(value)(items)
        except (TypeError, ValueError) as error:
            raise CudaGraphInputError(
                f"tuple type {type(value).__qualname__} cannot be reconstructed"
            ) from error
    if isinstance(value, list):
        return [make_static_tree(torch, item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: make_static_tree(torch, getattr(value, field.name))
                for field in fields(value)
            },
        )
    return value


def copy_into_static(torch: Any, static: Any, value: Any) -> None:
    if isinstance(static, torch.Tensor):
        if isinstance(value, DynamicScalar):
            static.fill_(value.value)
        else:
            static.copy_(value)
        return
    if isinstance(static, Mapping):
        for key in static:
            copy_into_static(torch, static[key], value[key])
        return
    if isinstance(static, (tuple, list)):
        for target, source in zip(static, value, strict=True):
            copy_into_static(torch, target, source)
        return
    if is_dataclass(static) and not isinstance(static, type):
        for field in fields(static):
            copy_into_static(
                torch,
                getattr(static, field.name),
                getattr(value, field.name),
            )


def clone_output_tree(torch: Any, value: Any) -> Any:
    """Detach returned values from graph-owned buffers."""

    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, Mapping):
        return mapping_like(
            value,
            [(key, clone_output_tree(torch, item)) for key, item in value.items()],
        )
    if isinstance(value, tuple):
        items = tuple(clone_output_tree(torch, item) for item in value)
        if type(value) is tuple:
            return items
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return type(value)(items)
    if isinstance(value, list):
        return [clone_output_tree(torch, item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: clone_output_tree(torch, getattr(value, field.name))
                for field in fields(value)
            },
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise CudaGraphInputError(
        f"output leaf type {type(value).__module__}.{type(value).__qualname__} is not supported"
    )
