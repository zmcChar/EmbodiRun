"""Dependency-free contracts for language-conditioned mobile navigation.

These values cross model, serving, and robot ownership boundaries.  They keep
sensor lineage and spatial output semantics explicit; converting a waypoint
into instantaneous base velocity remains a robot-owned operation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal, TypeAlias

from .types import Metadata

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_RGB_CONTEXT = 64
MAX_WAYPOINTS = 64
MAX_WAYPOINT_COORDINATE_M = 20.0
MAX_OUTPUT_VALID_FOR_S = 10.0

_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RGB_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class NavigationContractError(ValueError):
    """A navigation request or result violates its public contract."""


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NavigationContractError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise NavigationContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise NavigationContractError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise NavigationContractError(f"{name} must be at most {maximum}")
    return result


def _sequence(value: object, name: str = "sequence") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NavigationContractError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NavigationContractError(f"{name} must be a positive integer")
    return value


def _bytes(value: object, name: str, *, signature: bytes | None = None) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise NavigationContractError(f"{name} must be non-empty bytes")
    if len(value) > MAX_IMAGE_BYTES:
        raise NavigationContractError(f"{name} exceeds {MAX_IMAGE_BYTES} bytes")
    if signature is not None and not value.startswith(signature):
        raise NavigationContractError(f"{name} has an invalid file signature")
    return value


def _json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NavigationContractError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise NavigationContractError(f"{name} keys must be strings")
            _json_value(nested, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _json_value(nested, f"{name}[{index}]")
        return
    raise NavigationContractError(f"{name} must contain JSON-compatible values")


def _metadata(value: Metadata, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NavigationContractError(f"{name} must be a mapping")
    _json_value(value, name)
    return dict(value)


@dataclass(frozen=True, slots=True)
class EncodedRGBFrame:
    """One encoded RGB observation with authoritative capture lineage."""

    sequence: int
    captured_at_s: float
    data: bytes
    media_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _sequence(self.sequence, "rgb.sequence"))
        object.__setattr__(
            self,
            "captured_at_s",
            _finite(self.captured_at_s, "rgb.captured_at_s", minimum=0.0),
        )
        if self.media_type not in _RGB_MEDIA_TYPES:
            raise NavigationContractError(f"unsupported RGB media type: {self.media_type!r}")
        signature = b"\xff\xd8" if self.media_type == "image/jpeg" else None
        object.__setattr__(self, "data", _bytes(self.data, "rgb.data", signature=signature))
        if (self.width is None) != (self.height is None):
            raise NavigationContractError("RGB width and height must be provided together")
        if self.width is not None:
            object.__setattr__(self, "width", _positive_int(self.width, "rgb.width"))
            object.__setattr__(self, "height", _positive_int(self.height, "rgb.height"))


@dataclass(frozen=True, slots=True)
class EncodedDepthFrame:
    """Registered native uint16 depth encoded losslessly as PNG."""

    sequence: int
    captured_at_s: float
    data: bytes
    width: int
    height: int
    scale_m: float
    registered_to_rgb: bool = True
    encoding: Literal["uint16"] = "uint16"
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _sequence(self.sequence, "depth.sequence"))
        object.__setattr__(
            self,
            "captured_at_s",
            _finite(self.captured_at_s, "depth.captured_at_s", minimum=0.0),
        )
        object.__setattr__(
            self,
            "data",
            _bytes(self.data, "depth.data", signature=b"\x89PNG\r\n\x1a\n"),
        )
        object.__setattr__(self, "width", _positive_int(self.width, "depth.width"))
        object.__setattr__(self, "height", _positive_int(self.height, "depth.height"))
        object.__setattr__(
            self,
            "scale_m",
            _finite(
                self.scale_m,
                "depth.scale_m",
                minimum=math.nextafter(0.0, 1.0),
                maximum=0.1,
            ),
        )
        if self.encoding != "uint16":
            raise NavigationContractError("depth encoding must be 'uint16'")
        if self.registered_to_rgb is not True:
            raise NavigationContractError("depth must be registered to RGB")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "depth.metadata"))


@dataclass(frozen=True, slots=True)
class NavigationObservation:
    """Episode-scoped observation shared by Qwen, VLN, and navigation VLA."""

    episode_id: str
    sequence: int
    reset: bool
    rgb_frames: Sequence[EncodedRGBFrame]
    depth: EncodedDepthFrame | None = None
    robot_state: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not _EPISODE_ID.fullmatch(self.episode_id):
            raise NavigationContractError(
                "episode_id must be 1-128 characters using letters, digits, '.', '_', ':', or '-'"
            )
        sequence = _sequence(self.sequence, "observation.sequence")
        object.__setattr__(self, "sequence", sequence)
        if not isinstance(self.reset, bool):
            raise NavigationContractError("reset must be a boolean")
        if isinstance(self.rgb_frames, (str, bytes)) or not isinstance(self.rgb_frames, Sequence):
            raise NavigationContractError("rgb_frames must be a sequence")
        frames = tuple(self.rgb_frames)
        if not frames or len(frames) > MAX_RGB_CONTEXT:
            raise NavigationContractError(f"rgb_frames must contain 1-{MAX_RGB_CONTEXT} frames")
        if any(not isinstance(frame, EncodedRGBFrame) for frame in frames):
            raise NavigationContractError("rgb_frames must contain EncodedRGBFrame values")
        if any(
            current.captured_at_s <= previous.captured_at_s
            for previous, current in pairwise(frames)
        ):
            raise NavigationContractError("RGB capture timestamps must be strictly increasing")
        if frames[-1].sequence != sequence:
            raise NavigationContractError("latest RGB sequence must match observation sequence")
        object.__setattr__(self, "rgb_frames", frames)
        if self.depth is not None:
            if not isinstance(self.depth, EncodedDepthFrame):
                raise NavigationContractError("depth must be an EncodedDepthFrame or None")
            if self.depth.sequence != sequence:
                raise NavigationContractError("depth sequence must match observation sequence")
            if not math.isclose(
                self.depth.captured_at_s,
                frames[-1].captured_at_s,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise NavigationContractError("depth and latest RGB capture time must match")
            if frames[-1].width is not None and (self.depth.width, self.depth.height) != (
                frames[-1].width,
                frames[-1].height,
            ):
                raise NavigationContractError("registered RGB and depth dimensions must match")
        object.__setattr__(self, "robot_state", _metadata(self.robot_state, "robot_state"))
        object.__setattr__(self, "metadata", _metadata(self.metadata, "metadata"))

    @property
    def latest_rgb(self) -> EncodedRGBFrame:
        return self.rgb_frames[-1]


@dataclass(frozen=True, slots=True)
class NavigationRequest:
    instruction: str
    observation: NavigationObservation

    def __post_init__(self) -> None:
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise NavigationContractError("instruction must be a non-empty string")
        if len(self.instruction) > 10_000:
            raise NavigationContractError("instruction exceeds 10000 characters")
        if not isinstance(self.observation, NavigationObservation):
            raise NavigationContractError("observation must be a NavigationObservation")


@dataclass(frozen=True, slots=True)
class Waypoint:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x_m",
            _finite(
                self.x_m,
                "waypoint.x_m",
                minimum=-MAX_WAYPOINT_COORDINATE_M,
                maximum=MAX_WAYPOINT_COORDINATE_M,
            ),
        )
        object.__setattr__(
            self,
            "y_m",
            _finite(
                self.y_m,
                "waypoint.y_m",
                minimum=-MAX_WAYPOINT_COORDINATE_M,
                maximum=MAX_WAYPOINT_COORDINATE_M,
            ),
        )
        object.__setattr__(
            self,
            "yaw_rad",
            _finite(self.yaw_rad, "waypoint.yaw_rad", minimum=-math.pi, maximum=math.pi),
        )


@dataclass(frozen=True, slots=True)
class WaypointPlan:
    observation_sequence: int
    waypoints: Sequence[Waypoint]
    terminal: bool = False
    confidence: float = 1.0
    valid_for_s: float = 5.0
    frame: Literal["base_link"] = "base_link"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_sequence",
            _sequence(self.observation_sequence, "observation_sequence"),
        )
        if self.frame != "base_link":
            raise NavigationContractError("waypoint frame must be 'base_link'")
        if isinstance(self.waypoints, (str, bytes)) or not isinstance(self.waypoints, Sequence):
            raise NavigationContractError("waypoints must be a sequence")
        waypoints = tuple(self.waypoints)
        if not waypoints and not self.terminal:
            raise NavigationContractError("an empty waypoint plan must be terminal")
        if len(waypoints) > MAX_WAYPOINTS:
            raise NavigationContractError(f"waypoint plan exceeds {MAX_WAYPOINTS} points")
        if any(not isinstance(waypoint, Waypoint) for waypoint in waypoints):
            raise NavigationContractError("waypoints must contain Waypoint values")
        if not isinstance(self.terminal, bool):
            raise NavigationContractError("terminal must be a boolean")
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(
            self,
            "confidence",
            _finite(self.confidence, "confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "valid_for_s",
            _finite(
                self.valid_for_s,
                "valid_for_s",
                minimum=math.nextafter(0.0, 1.0),
                maximum=MAX_OUTPUT_VALID_FOR_S,
            ),
        )


NavigationResult: TypeAlias = WaypointPlan


__all__ = [
    "EncodedDepthFrame",
    "EncodedRGBFrame",
    "NavigationContractError",
    "NavigationObservation",
    "NavigationRequest",
    "NavigationResult",
    "Waypoint",
    "WaypointPlan",
]
