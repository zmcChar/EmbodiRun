"""Structural and numeric validation of OpenPI action outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import OpenPIDependencyError, OpenPIProtocolError


def validate_openpi_actions(
    actions: Any,
    *,
    expected_batch_size: int | None = 1,
    expected_action_horizon: int | None = None,
    expected_action_dim: int | None = None,
) -> Any:
    """Validate and return an OpenPI action array or named action mapping.

    Arrays may be unbatched ``[horizon, dim]`` or batched
    ``[batch, horizon, dim]``. Every array in a multi-head action mapping must
    share the same normalized ``(batch, horizon)``. ``expected_action_dim``
    denotes the final dimension for a single array and the sum of final
    dimensions for a named mapping.
    """

    numpy = _load_numpy()
    if expected_batch_size is not None and expected_batch_size <= 0:
        raise ValueError("expected_batch_size must be greater than zero or None")
    if expected_action_horizon is not None and expected_action_horizon <= 0:
        raise ValueError("expected_action_horizon must be greater than zero or None")
    if expected_action_dim is not None and expected_action_dim <= 0:
        raise ValueError("expected_action_dim must be greater than zero or None")

    if isinstance(actions, numpy.ndarray):
        action_items = (("actions", actions),)
        validated: Any = actions
    elif isinstance(actions, Mapping):
        if not actions:
            raise OpenPIProtocolError("action mapping must not be empty")
        copied: dict[str, Any] = {}
        for name, array in actions.items():
            if not isinstance(name, str) or not name:
                raise OpenPIProtocolError("action mapping keys must be non-empty strings")
            copied[name] = array
        action_items = tuple(copied.items())
        validated = copied
    else:
        raise OpenPIProtocolError(
            "OpenPI response actions must be a numpy.ndarray or a mapping of arrays"
        )

    common_batch_horizon: tuple[int, int] | None = None
    total_action_dim = 0
    for name, array in action_items:
        batch, horizon, action_dim = _action_signature(array, name=name, numpy=numpy)
        signature = (batch, horizon)
        if common_batch_horizon is None:
            common_batch_horizon = signature
        elif signature != common_batch_horizon:
            raise OpenPIProtocolError(
                "all action heads must share batch and horizon dimensions; "
                f"expected {common_batch_horizon}, action {name!r} has {signature}"
            )
        total_action_dim += action_dim

    assert common_batch_horizon is not None
    batch, horizon = common_batch_horizon
    if expected_batch_size is not None and batch != expected_batch_size:
        raise OpenPIProtocolError(f"expected action batch size {expected_batch_size}, got {batch}")
    if expected_action_horizon is not None and horizon != expected_action_horizon:
        raise OpenPIProtocolError(
            f"expected action horizon {expected_action_horizon}, got {horizon}"
        )
    if expected_action_dim is not None and total_action_dim != expected_action_dim:
        raise OpenPIProtocolError(
            f"expected total action dimension {expected_action_dim}, got {total_action_dim}"
        )
    return validated


def _load_numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise OpenPIDependencyError(
            "OpenPI action validation requires the optional `numpy` package"
        ) from error
    return numpy


def _action_signature(array: Any, *, name: str, numpy: Any) -> tuple[int, int, int]:
    if not isinstance(array, numpy.ndarray):
        raise OpenPIProtocolError(
            f"action {name!r} must decode to numpy.ndarray, got {type(array).__name__}"
        )
    if array.dtype != numpy.dtype(numpy.float32):
        raise OpenPIProtocolError(f"action {name!r} must have dtype float32, got {array.dtype}")
    if array.ndim not in {2, 3}:
        raise OpenPIProtocolError(
            f"action {name!r} must have shape [horizon, dim] or "
            f"[batch, horizon, dim], got {array.shape}"
        )
    if any(int(size) <= 0 for size in array.shape):
        raise OpenPIProtocolError(f"action {name!r} has an empty dimension: {array.shape}")
    if not bool(numpy.isfinite(array).all()):
        raise OpenPIProtocolError(f"action {name!r} contains non-finite values")

    if array.ndim == 2:
        horizon, action_dim = (int(size) for size in array.shape)
        return 1, horizon, action_dim
    batch, horizon, action_dim = (int(size) for size in array.shape)
    return batch, horizon, action_dim
