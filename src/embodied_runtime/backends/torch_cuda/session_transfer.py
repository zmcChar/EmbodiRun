"""Tensor-tree transfer and device-local state arithmetic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from typing import Any


def move_tensor_tree(
    torch: Any,
    value: Any,
    *,
    device: Any,
    dtype: Any | None,
    non_blocking: bool,
) -> Any:
    """Recursively move tensors while preserving the surrounding tree shape."""

    if isinstance(value, torch.Tensor):
        target_dtype = dtype if dtype is not None and value.is_floating_point() else None
        return value.to(
            device=device,
            dtype=target_dtype,
            non_blocking=non_blocking,
        )
    if isinstance(value, Mapping):
        return {
            key: move_tensor_tree(
                torch,
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        moved = tuple(
            move_tensor_tree(
                torch,
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for item in value
        )
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return moved
    if isinstance(value, list):
        return [
            move_tensor_tree(
                torch,
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for item in value
        ]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: move_tensor_tree(
                torch,
                getattr(value, field.name),
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for field in fields(value)
        }
        return replace(value, **updates)
    return value


def add_scaled_tree(torch: Any, state: Any, update: Any, scale: float) -> Any:
    """Apply ``state + scale * update`` without moving arithmetic off-device."""

    if isinstance(state, Mapping):
        if not isinstance(update, Mapping) or state.keys() != update.keys():
            raise TypeError("state and update mappings must have identical keys")
        return type(state)(
            (key, add_scaled_tree(torch, state[key], update[key], scale)) for key in state
        )
    if isinstance(state, tuple):
        if not isinstance(update, tuple) or len(state) != len(update):
            raise TypeError("state and update tuples must have identical lengths")
        values = tuple(
            add_scaled_tree(torch, left, right, scale) for left, right in zip(state, update)
        )
        if hasattr(state, "_fields"):
            return type(state)(*values)
        return values
    if isinstance(state, list):
        if not isinstance(update, list) or len(state) != len(update):
            raise TypeError("state and update lists must have identical lengths")
        return [add_scaled_tree(torch, left, right, scale) for left, right in zip(state, update)]
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
        if not isinstance(update, type(state)) or len(state) != len(update):
            raise TypeError("state and update sequences must have identical lengths")
        return type(state)(
            add_scaled_tree(torch, left, right, scale) for left, right in zip(state, update)
        )
    if is_dataclass(state) and not isinstance(state, type):
        if not is_dataclass(update) or type(update) is not type(state):
            raise TypeError("state and update dataclasses must have identical types")
        return replace(
            state,
            **{
                field.name: add_scaled_tree(
                    torch,
                    getattr(state, field.name),
                    getattr(update, field.name),
                    scale,
                )
                for field in fields(state)
            },
        )
    try:
        if isinstance(state, torch.Tensor) and isinstance(update, torch.Tensor):
            # Preserve the engine's declared Euler operation order exactly.
            return state + update * scale
        return state + update * scale
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(
            f"cannot add scaled leaves {type(state).__name__} and {type(update).__name__}"
        ) from error
