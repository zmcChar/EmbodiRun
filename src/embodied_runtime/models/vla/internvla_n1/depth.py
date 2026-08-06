"""Registered uint16 PNG depth conversion for InternVLA-N1 NavDP."""

from __future__ import annotations

import importlib
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_DEPTH_SCALE_M = 0.1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class InternVLADepthError(ValueError):
    """A NavDP depth sample is ambiguous, malformed, or misregistered."""


@dataclass(frozen=True, slots=True)
class InternVLADepthPayload:
    """Dependency-neutral encoded depth fields consumed by this model package."""

    data: bytes
    width: int
    height: int
    scale_m: float
    registered_to_rgb: bool = True
    encoding: str = "uint16"


def _field(payload: object, name: str) -> object:
    if isinstance(payload, Mapping):
        try:
            return payload[name]
        except KeyError as error:
            raise InternVLADepthError(f"depth payload is missing {name!r}") from error
    try:
        return getattr(payload, name)
    except AttributeError as error:
        raise InternVLADepthError(f"depth payload is missing {name!r}") from error


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InternVLADepthError(f"{name} must be a positive integer")
    if value > MAX_IMAGE_DIMENSION:
        raise InternVLADepthError(f"{name} exceeds {MAX_IMAGE_DIMENSION}")
    return value


def _scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InternVLADepthError("depth.scale_m must be a number")
    result = float(value)
    if not math.isfinite(result) or not (0.0 < result <= MAX_DEPTH_SCALE_M):
        raise InternVLADepthError(f"depth.scale_m must be finite and in (0, {MAX_DEPTH_SCALE_M}]")
    return result


def normalize_depth_meters(value: object, *, expected_shape: tuple[int, int]) -> Any:
    """Return contiguous float32 metres after validating a registered array."""

    try:
        numpy = importlib.import_module("numpy")
    except ImportError as error:  # pragma: no cover - InternNav requires NumPy
        raise InternVLADepthError("depth conversion requires NumPy") from error
    try:
        array = numpy.asarray(value)
    except Exception as error:
        raise InternVLADepthError(f"cannot convert depth to an array: {error}") from error
    if array.ndim != 2:
        raise InternVLADepthError("depth must be a two-dimensional array")
    if tuple(array.shape) != tuple(expected_shape):
        raise InternVLADepthError("registered depth dimensions must match RGB")
    if array.dtype.kind not in "uif":
        raise InternVLADepthError("depth metres must have a numeric dtype")
    try:
        result = numpy.ascontiguousarray(array, dtype=numpy.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise InternVLADepthError(f"cannot convert depth to float32 metres: {error}") from error
    if not bool(numpy.isfinite(result).all()):
        raise InternVLADepthError("depth metres must contain only finite values")
    if bool((result < 0.0).any()):
        raise InternVLADepthError("depth metres must not contain negative values")
    return result


def decode_depth_payload(
    payload: InternVLADepthPayload | Mapping[str, object] | object,
    *,
    expected_shape: tuple[int, int],
) -> Any:
    """Decode native uint16 PNG samples and apply their explicit metre scale.

    The payload may be the internal dataclass, a mapping, or a shared contract
    object exposing the same attributes. No undocumented ``metres * 10000``
    convention is accepted.
    """

    if _field(payload, "encoding") != "uint16":
        raise InternVLADepthError("depth encoding must be 'uint16'")
    if _field(payload, "registered_to_rgb") is not True:
        raise InternVLADepthError("depth must be registered to RGB")
    width = _positive_integer(_field(payload, "width"), "depth.width")
    height = _positive_integer(_field(payload, "height"), "depth.height")
    scale_m = _scale(_field(payload, "scale_m"))
    data = _field(payload, "data")
    if not isinstance(data, bytes) or not data:
        raise InternVLADepthError("depth.data must be non-empty bytes")
    if len(data) > MAX_IMAGE_BYTES:
        raise InternVLADepthError(f"depth.data exceeds {MAX_IMAGE_BYTES} bytes")
    if not data.startswith(PNG_SIGNATURE):
        raise InternVLADepthError("depth.data must contain a PNG image")

    try:
        numpy = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(io.BytesIO(data)) as image:
            image.load()
            array = numpy.asarray(image)
    except Exception as error:
        raise InternVLADepthError(f"cannot decode depth PNG: {error}") from error
    if array.ndim != 2:
        raise InternVLADepthError("depth PNG must have one channel")
    if array.dtype.kind != "u" or array.dtype.itemsize != 2:
        raise InternVLADepthError("depth PNG must preserve native uint16 samples")
    if tuple(array.shape) != (height, width):
        raise InternVLADepthError("depth PNG dimensions do not match the declared width and height")
    if tuple(array.shape) != tuple(expected_shape):
        raise InternVLADepthError("registered depth dimensions must match RGB")
    metres = array.astype(numpy.float32) * scale_m
    return normalize_depth_meters(metres, expected_shape=expected_shape)


__all__ = [
    "MAX_DEPTH_SCALE_M",
    "InternVLADepthError",
    "InternVLADepthPayload",
    "decode_depth_payload",
    "normalize_depth_meters",
]
