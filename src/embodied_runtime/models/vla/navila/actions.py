"""NaVILA's native one-action textual navigation vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .errors import NaVILANativeOutputError

MAX_FORWARD_CM = 75
MAX_TURN_DEG = 45
FORWARD_INCREMENT_CM = 25
TURN_INCREMENT_DEG = 15
FORWARD_MAGNITUDES_CM = (25, 50, 75)
TURN_MAGNITUDES_DEG = (15, 30, 45)


class NaVILAPrimitive(str, Enum):
    STOP = "stop"
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


@dataclass(frozen=True, slots=True)
class NaVILAAction:
    primitive: NaVILAPrimitive
    magnitude: int | None = None
    unit: Literal["cm", "degree"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, NaVILAPrimitive):
            raise NaVILANativeOutputError("primitive must be a NaVILAPrimitive")
        if self.primitive is NaVILAPrimitive.STOP:
            if self.magnitude is not None or self.unit is not None:
                raise NaVILANativeOutputError("STOP must not carry a magnitude")
            return
        if isinstance(self.magnitude, bool) or not isinstance(self.magnitude, int):
            raise NaVILANativeOutputError("motion actions require an integer magnitude")
        expected_unit = "cm" if self.primitive is NaVILAPrimitive.MOVE_FORWARD else "degree"
        if self.unit != expected_unit:
            raise NaVILANativeOutputError(f"{self.primitive.value} requires {expected_unit}")
        allowed = (
            FORWARD_MAGNITUDES_CM
            if self.primitive is NaVILAPrimitive.MOVE_FORWARD
            else TURN_MAGNITUDES_DEG
        )
        if self.magnitude not in allowed:
            raise NaVILANativeOutputError(
                f"{self.primitive.value} magnitude must be one of {allowed}"
            )


_PRIMITIVES = (
    (NaVILAPrimitive.STOP, re.compile(r"\bstop\b", re.IGNORECASE)),
    (NaVILAPrimitive.MOVE_FORWARD, re.compile(r"\bmove\s+forward\b", re.IGNORECASE)),
    (NaVILAPrimitive.TURN_LEFT, re.compile(r"\bturn\s+left\b", re.IGNORECASE)),
    (NaVILAPrimitive.TURN_RIGHT, re.compile(r"\bturn\s+right\b", re.IGNORECASE)),
)
_FORWARD_MAGNITUDE = re.compile(r"\bmove\s+forward\s+(\d+)\s*(?:cm|centimeters?)\b", re.IGNORECASE)
_LEFT_MAGNITUDE = re.compile(r"\bturn\s+left\s+(\d+)\s*degrees?\b", re.IGNORECASE)
_RIGHT_MAGNITUDE = re.compile(r"\bturn\s+right\s+(\d+)\s*degrees?\b", re.IGNORECASE)


def _magnitude(
    text: str,
    pattern: re.Pattern[str],
    *,
    default: int,
    allowed: tuple[int, ...],
) -> int:
    captures = pattern.findall(text)
    if len(captures) > 1:
        raise NaVILANativeOutputError("decoded text contains multiple action magnitudes")
    value = default if not captures else int(captures[0])
    if value not in allowed:
        raise NaVILANativeOutputError(f"action magnitude must be one of {allowed}")
    return value


def parse_navila_action(text: str) -> NaVILAAction:
    """Parse exactly one official action without silently changing its magnitude."""

    if not isinstance(text, str) or not text.strip():
        raise NaVILANativeOutputError("decoded text must be non-empty")
    if len(text) > 4096:
        raise NaVILANativeOutputError("decoded text exceeds 4096 characters")
    matches = [primitive for primitive, pattern in _PRIMITIVES for _ in pattern.finditer(text)]
    if len(matches) != 1:
        raise NaVILANativeOutputError("decoded text must contain exactly one NaVILA action")
    primitive = matches[0]
    if primitive is NaVILAPrimitive.STOP:
        return NaVILAAction(primitive)
    if primitive is NaVILAPrimitive.MOVE_FORWARD:
        value = _magnitude(
            text,
            _FORWARD_MAGNITUDE,
            default=FORWARD_INCREMENT_CM,
            allowed=FORWARD_MAGNITUDES_CM,
        )
        return NaVILAAction(primitive, value, "cm")
    value = _magnitude(
        text,
        _LEFT_MAGNITUDE if primitive is NaVILAPrimitive.TURN_LEFT else _RIGHT_MAGNITUDE,
        default=TURN_INCREMENT_DEG,
        allowed=TURN_MAGNITUDES_DEG,
    )
    return NaVILAAction(primitive, value, "degree")


__all__ = [
    "FORWARD_INCREMENT_CM",
    "FORWARD_MAGNITUDES_CM",
    "MAX_FORWARD_CM",
    "MAX_TURN_DEG",
    "TURN_INCREMENT_DEG",
    "TURN_MAGNITUDES_DEG",
    "NaVILAAction",
    "NaVILAPrimitive",
    "parse_navila_action",
]
