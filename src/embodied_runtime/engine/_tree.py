"""Operations on opaque tensor trees without importing a tensor framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def split_batch(value: Any, batch_size: int) -> list[Any]:
    """Best-effort split of a batched output into per-request tensor trees."""

    if batch_size == 1:
        return [_index_or_identity(value, 0, batch_size)]
    if isinstance(value, Mapping):
        split_fields = {key: split_batch(item, batch_size) for key, item in value.items()}
        return [
            type(value)((key, parts[index]) for key, parts in split_fields.items())
            for index in range(batch_size)
        ]
    if _is_sequence(value) and len(value) == batch_size:
        return list(value)
    if _has_batch_axis(value, batch_size):
        return [value[index] for index in range(batch_size)]
    # Immutable metadata/scalars describe the complete batch and are repeated.
    return [value for _ in range(batch_size)]


def _index_or_identity(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, Mapping):
        return type(value)(
            (key, _index_or_identity(item, index, batch_size)) for key, item in value.items()
        )
    if _has_batch_axis(value, batch_size):
        return value[index]
    return value


def _has_batch_axis(value: Any, batch_size: int) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    try:
        return len(shape) > 0 and int(shape[0]) == batch_size
    except (TypeError, ValueError):
        return False


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
