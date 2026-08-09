"""Stable numeric OpenPI wire schema for navigation-native predictions."""

from __future__ import annotations

from typing import Any

from embodied_runtime.models.vla.navila import NaVILAAction, NaVILAPrimitive
from embodied_runtime.models.vln.streamvln import (
    MAX_FUTURE_ACTIONS,
    normalize_native_actions,
)

STREAMVLN_ACTION_HEAD = "discrete"
NAVILA_ACTION_HEAD = "navigation"
STREAMVLN_PADDING_ID = -1

_NAVILA_TO_ID = {
    NaVILAPrimitive.STOP: 0,
    NaVILAPrimitive.MOVE_FORWARD: 1,
    NaVILAPrimitive.TURN_LEFT: 2,
    NaVILAPrimitive.TURN_RIGHT: 3,
}
_ID_TO_NAVILA = {value: key for key, value in _NAVILA_TO_ID.items()}


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as error:  # pragma: no cover - optional deployment dependency
        raise RuntimeError("vLLM-Omni navigation transport requires numpy") from error
    return numpy


def _wire_array(actions: Any, head: str) -> Any:
    if not isinstance(actions, dict) or head not in actions:
        raise ValueError(f"vLLM-Omni response must contain action head {head!r}")
    numpy = _numpy()
    values = numpy.asarray(actions[head])
    if values.dtype != numpy.float32:
        raise ValueError(f"vLLM-Omni action head {head!r} must use float32")
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(f"vLLM-Omni action head {head!r} must have batch size 1")
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"vLLM-Omni action head {head!r} must have rank 2 or 3")
    if not bool(numpy.isfinite(values).all()):
        raise ValueError(f"vLLM-Omni action head {head!r} must contain finite values")
    return values


def encode_streamvln_actions(actions: Any) -> dict[str, Any]:
    """Encode at most four native ids as a fixed OpenPI action tensor."""

    numpy = _numpy()
    normalized = normalize_native_actions(actions, max_actions=MAX_FUTURE_ACTIONS)
    padded = numpy.full(
        (1, MAX_FUTURE_ACTIONS, 1),
        STREAMVLN_PADDING_ID,
        dtype=numpy.float32,
    )
    padded[0, : len(normalized), 0] = normalized
    return {STREAMVLN_ACTION_HEAD: padded}


def decode_streamvln_actions(actions: Any) -> tuple[int, ...]:
    values = _wire_array(actions, STREAMVLN_ACTION_HEAD)
    if values.shape != (MAX_FUTURE_ACTIONS, 1):
        raise ValueError(
            "vLLM-Omni StreamVLN actions must have shape "
            f"[{MAX_FUTURE_ACTIONS}, 1], got {tuple(values.shape)}"
        )
    flattened = values[:, 0]
    rounded = flattened.round()
    numpy = _numpy()
    if not bool(numpy.allclose(flattened, rounded, atol=0.0, rtol=0.0)):
        raise ValueError("vLLM-Omni StreamVLN actions must be integer-valued")
    action_ids = tuple(int(value) for value in rounded.tolist())
    try:
        padding_index = action_ids.index(STREAMVLN_PADDING_ID)
    except ValueError:
        padding_index = len(action_ids)
    if any(value != STREAMVLN_PADDING_ID for value in action_ids[padding_index:]):
        raise ValueError("vLLM-Omni StreamVLN padding must be a contiguous suffix")
    return normalize_native_actions(action_ids[:padding_index], max_actions=MAX_FUTURE_ACTIONS)


def encode_navila_action(action: NaVILAAction) -> dict[str, Any]:
    """Encode primitive, magnitude, and unit into one fixed float32 vector."""

    if not isinstance(action, NaVILAAction):
        raise TypeError("action must be a NaVILAAction")
    numpy = _numpy()
    unit_id = {None: 0, "cm": 1, "degree": 2}[action.unit]
    values = numpy.asarray(
        [[[_NAVILA_TO_ID[action.primitive], action.magnitude or 0, unit_id]]],
        dtype=numpy.float32,
    )
    return {NAVILA_ACTION_HEAD: values}


def decode_navila_action(actions: Any) -> NaVILAAction:
    values = _wire_array(actions, NAVILA_ACTION_HEAD)
    if values.shape != (1, 3):
        raise ValueError(
            f"vLLM-Omni NaVILA actions must have shape [1, 3], got {tuple(values.shape)}"
        )
    row = values[0]
    rounded = row.round()
    numpy = _numpy()
    if not bool(numpy.allclose(row, rounded, atol=0.0, rtol=0.0)):
        raise ValueError("vLLM-Omni NaVILA action fields must be integer-valued")
    primitive_id, magnitude, unit_id = (int(value) for value in rounded.tolist())
    try:
        primitive = _ID_TO_NAVILA[primitive_id]
    except KeyError as error:
        raise ValueError(f"unknown vLLM-Omni NaVILA primitive id {primitive_id}") from error
    if primitive is NaVILAPrimitive.STOP:
        if magnitude != 0 or unit_id != 0:
            raise ValueError("vLLM-Omni NaVILA STOP must have zero magnitude and unit")
        return NaVILAAction(primitive)
    expected_unit_id = 1 if primitive is NaVILAPrimitive.MOVE_FORWARD else 2
    if unit_id != expected_unit_id:
        raise ValueError("vLLM-Omni NaVILA action has an incompatible unit id")
    unit = "cm" if unit_id == 1 else "degree"
    return NaVILAAction(primitive, magnitude, unit)


__all__ = [
    "NAVILA_ACTION_HEAD",
    "STREAMVLN_ACTION_HEAD",
    "STREAMVLN_PADDING_ID",
    "decode_navila_action",
    "decode_streamvln_actions",
    "encode_navila_action",
    "encode_streamvln_actions",
]
