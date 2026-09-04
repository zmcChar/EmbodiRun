"""Persistent color-only RealSense camera source."""

from __future__ import annotations

import importlib
import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..camera import CameraFrame


class RealSenseCameraError(RuntimeError):
    """A configured RealSense camera could not produce a color frame."""


@dataclass(frozen=True, slots=True)
class RealSenseCameraConfig:
    name: str
    serial: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    jpeg_quality: int = 90
    frame_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        for name in ("name", "serial"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"camera {name} must not be empty")
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if (
            isinstance(self.frame_timeout_s, bool)
            or not isinstance(self.frame_timeout_s, (int, float))
            or not math.isfinite(self.frame_timeout_s)
            or self.frame_timeout_s <= 0
        ):
            raise ValueError("frame_timeout_s must be a positive number")


class RealSenseCameraSource:
    """Open each RealSense pipeline once and reuse it for every capture."""

    def __init__(
        self,
        cameras: Sequence[RealSenseCameraConfig],
        *,
        rs_module: Any | None = None,
        jpeg_encoder: Callable[[bytes, int, int, int], bytes] | None = None,
    ) -> None:
        self._cameras = tuple(cameras)
        if not self._cameras:
            raise RealSenseCameraError("at least one RealSense camera is required")
        if rs_module is None:
            try:
                rs_module = importlib.import_module("pyrealsense2")
            except ImportError as error:
                raise RealSenseCameraError(
                    "RealSense capture requires pyrealsense2 in the robot environment"
                ) from error
        self._rs = rs_module
        self._jpeg_encoder = jpeg_encoder or _encode_rgb_as_jpeg
        self._pipelines: list[Any] = []
        try:
            for camera in self._cameras:
                pipeline = rs_module.pipeline()
                config = rs_module.config()
                config.enable_device(camera.serial)
                config.enable_stream(
                    rs_module.stream.color,
                    camera.width,
                    camera.height,
                    rs_module.format.rgb8,
                    camera.fps,
                )
                self._pipelines.append(pipeline)
                pipeline.start(config)
        except Exception as error:
            self.close()
            if isinstance(error, RealSenseCameraError):
                raise
            raise RealSenseCameraError(
                f"could not start RealSense color pipeline: {error}"
            ) from error

    def capture(self) -> tuple[CameraFrame, ...]:
        images: list[CameraFrame] = []
        for camera, pipeline in zip(self._cameras, self._pipelines):
            try:
                frames = pipeline.wait_for_frames(
                    max(1, int(camera.frame_timeout_s * 1000))
                )
                color = frames.get_color_frame()
            except RuntimeError as error:
                raise RealSenseCameraError(
                    f"camera {camera.name!r} frame capture failed: {error}"
                ) from error
            if not color:
                raise RealSenseCameraError(
                    f"camera {camera.name!r} returned no color frame"
                )
            encoded = self._jpeg_encoder(
                bytes(color.get_data()),
                camera.width,
                camera.height,
                camera.jpeg_quality,
            )
            images.append(CameraFrame(camera.name, "image/jpeg", encoded))
        return tuple(images)

    def close(self) -> None:
        for pipeline in reversed(self._pipelines):
            try:
                pipeline.stop()
            except Exception:
                pass
        self._pipelines.clear()


def _encode_rgb_as_jpeg(raw: bytes, width: int, height: int, quality: int) -> bytes:
    expected = width * height * 3
    if len(raw) != expected:
        raise RealSenseCameraError(
            f"unexpected RGB8 frame size: got {len(raw)}, expected {expected}"
        )
    try:
        from PIL import Image
    except ImportError as error:
        raise RealSenseCameraError(
            "RealSense JPEG encoding requires Pillow in the robot environment"
        ) from error
    image = Image.frombytes("RGB", (width, height), raw)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=quality)
    return encoded.getvalue()


__all__ = [
    "RealSenseCameraConfig",
    "RealSenseCameraError",
    "RealSenseCameraSource",
]
