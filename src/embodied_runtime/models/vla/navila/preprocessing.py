"""Dependency-injected image batching for NaVILA's fixed eight-frame input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import NaVILAInferenceError
from .history import NAVILA_FRAME_COUNT

NAVILA_IMAGE_SIZE = 384
NAVILA_PADDING_SIZE = 512
EXPECTED_IMAGE_SHAPE = (8, 3, 384, 384)


def _exact_processor_size(value: object) -> bool:
    if isinstance(value, Mapping):
        return value.get("height") == NAVILA_IMAGE_SIZE and value.get("width") == NAVILA_IMAGE_SIZE
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and tuple(value) == (NAVILA_IMAGE_SIZE, NAVILA_IMAGE_SIZE)
    )


def _validate_processor(image_processor: Any) -> None:
    candidates = (
        getattr(image_processor, "size", None),
        getattr(image_processor, "crop_size", None),
    )
    if not any(_exact_processor_size(candidate) for candidate in candidates):
        raise NaVILAInferenceError("NaVILA image processor must use 384x384 inputs")


def _shape(value: Any, label: str) -> None:
    try:
        actual = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise NaVILAInferenceError(f"{label} must expose a tensor shape") from error
    if actual != EXPECTED_IMAGE_SHAPE:
        raise NaVILAInferenceError(f"{label} shape must be {EXPECTED_IMAGE_SHAPE}, got {actual}")


def _pil_images(rgb_frames: Sequence[Any], image_module: Any) -> list[Any]:
    if isinstance(rgb_frames, (str, bytes)) or not isinstance(rgb_frames, Sequence):
        raise NaVILAInferenceError("rgb_frames must be a sequence")
    if not 1 <= len(rgb_frames) <= NAVILA_FRAME_COUNT:
        raise NaVILAInferenceError("NaVILA requires between one and eight sampled RGB frames")
    images: list[Any] = []
    for frame in rgb_frames:
        try:
            if len(frame.shape) != 3 or int(frame.shape[2]) != 3:
                raise ValueError
            images.append(image_module.fromarray(frame).convert("RGB"))
        except Exception as error:
            raise NaVILAInferenceError("RGB frames must be HxWx3 uint8 arrays") from error
    padding = [
        image_module.new("RGB", (NAVILA_PADDING_SIZE, NAVILA_PADDING_SIZE), (0, 0, 0))
        for _ in range(NAVILA_FRAME_COUNT - len(images))
    ]
    return padding + images


def prepare_navila_images(
    rgb_frames: Sequence[Any],
    *,
    image_processor: Any,
    model_config: Any,
    process_images: Any,
    image_module: Any,
    device: Any,
    torch_dtype: Any,
) -> Any:
    """Left-pad history and run the official processor exactly once."""

    _validate_processor(image_processor)
    if not callable(process_images):
        raise NaVILAInferenceError("NaVILA process_images must be callable")
    images = _pil_images(rgb_frames, image_module)
    try:
        tensor = process_images(images, image_processor, model_config)
    except Exception as error:
        raise NaVILAInferenceError("NaVILA image preprocessing failed") from error
    _shape(tensor, "processed images")
    try:
        converted = tensor.to(device, dtype=torch_dtype)
    except Exception as error:
        raise NaVILAInferenceError("cannot move NaVILA images to the model device") from error
    _shape(converted, "converted images")
    return converted


__all__ = [
    "EXPECTED_IMAGE_SHAPE",
    "NAVILA_IMAGE_SIZE",
    "NAVILA_PADDING_SIZE",
    "prepare_navila_images",
]
