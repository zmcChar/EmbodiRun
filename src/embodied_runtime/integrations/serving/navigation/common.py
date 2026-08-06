"""Shared state and image handling for stateful local navigation providers."""

from __future__ import annotations

import importlib
import io
from dataclasses import dataclass
from typing import Any

from embodied_runtime.contracts import EncodedRGBFrame, NavigationObservation, NavigationRequest


class NavigationProviderError(RuntimeError):
    """A local navigation provider cannot safely advance its model state."""


def navigation_request(payload: object) -> NavigationRequest:
    if not isinstance(payload, NavigationRequest):
        raise TypeError("navigation provider payload must be a NavigationRequest")
    return payload


def decode_rgb(frame: EncodedRGBFrame) -> Any:
    """Decode one contract frame into a contiguous HxWx3 uint8 RGB array."""

    try:
        image_module = importlib.import_module("PIL.Image")
        numpy = importlib.import_module("numpy")
    except ImportError as error:  # pragma: no cover - optional deployment dependency
        raise NavigationProviderError("RGB decoding requires Pillow and NumPy") from error
    try:
        with image_module.open(io.BytesIO(frame.data)) as image:
            rgb = image.convert("RGB")
            array = numpy.asarray(rgb, dtype=numpy.uint8).copy(order="C")
    except Exception as error:
        raise NavigationProviderError(
            f"cannot decode {frame.media_type} RGB frame: {error}"
        ) from error
    actual_height, actual_width = int(array.shape[0]), int(array.shape[1])
    if frame.width is not None and (actual_width, actual_height) != (frame.width, frame.height):
        raise NavigationProviderError(
            "decoded RGB dimensions do not match EncodedRGBFrame metadata"
        )
    return array


@dataclass(slots=True)
class EpisodeCursor:
    """Validate one recurrent model's episode and observation ordering."""

    episode_id: str | None = None
    sequence: int | None = None

    def requires_reset(self, observation: NavigationObservation) -> bool:
        if self.episode_id is None:
            return True
        if observation.episode_id != self.episode_id:
            if not observation.reset:
                raise NavigationProviderError("a new episode_id must set observation.reset=true")
            return True
        if observation.reset:
            return True
        if self.sequence is not None and observation.sequence <= self.sequence:
            raise NavigationProviderError(
                f"observation sequence must increase beyond {self.sequence}"
            )
        return False

    def commit(self, observation: NavigationObservation) -> None:
        self.episode_id = observation.episode_id
        self.sequence = observation.sequence

    def clear(self) -> None:
        self.episode_id = None
        self.sequence = None


__all__ = [
    "EpisodeCursor",
    "NavigationProviderError",
    "decode_rgb",
    "navigation_request",
]
