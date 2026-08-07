"""Image decoding at the navigation-policy boundary."""

from __future__ import annotations

import importlib
import io
from typing import Any

from embodied_runtime.tasks.navigation import EncodedRGBFrame

from .errors import NavigationPolicyError


def decode_rgb(frame: EncodedRGBFrame) -> Any:
    """Decode one encoded frame into a contiguous HxWx3 uint8 RGB array."""

    try:
        image_module = importlib.import_module("PIL.Image")
        numpy = importlib.import_module("numpy")
    except ImportError as error:  # pragma: no cover - optional deployment dependency
        raise NavigationPolicyError("RGB decoding requires Pillow and NumPy") from error
    try:
        with image_module.open(io.BytesIO(frame.data)) as image:
            rgb = image.convert("RGB")
            array = numpy.asarray(rgb, dtype=numpy.uint8).copy(order="C")
    except Exception as error:
        raise NavigationPolicyError(
            f"cannot decode {frame.media_type} RGB frame: {error}"
        ) from error
    actual_height, actual_width = int(array.shape[0]), int(array.shape[1])
    if frame.width is not None and (actual_width, actual_height) != (
        frame.width,
        frame.height,
    ):
        raise NavigationPolicyError("decoded RGB dimensions do not match EncodedRGBFrame metadata")
    return array


__all__ = ["decode_rgb"]
