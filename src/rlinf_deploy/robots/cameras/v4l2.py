"""V4L2 camera capture through OpenCV."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .camera import CameraFrame


class CameraError(RuntimeError):
    """A configured camera could not produce a valid frame."""


@dataclass(frozen=True, slots=True)
class V4L2CameraConfig:
    name: str
    device: str
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera name must not be empty")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("camera device must not be empty")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("camera fps must be a finite positive number")


class V4L2CameraSource:
    """Capture synchronized-enough JPEG frames from V4L2 cameras."""

    def __init__(
        self,
        cameras: Sequence[V4L2CameraConfig],
        *,
        cv2_module: Any | None = None,
    ) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise CameraError(
                    "V4L2 capture requires OpenCV in the robot environment"
                ) from error
        self._cv2 = cv2_module
        self._cameras = tuple(cameras)
        if not self._cameras:
            raise CameraError("at least one V4L2 camera is required")
        self._captures: list[Any] = []
        try:
            for camera in self._cameras:
                capture = cv2_module.VideoCapture(camera.device, cv2_module.CAP_V4L2)
                self._captures.append(capture)
                if not capture.isOpened():
                    raise CameraError(
                        f"camera {camera.name!r} cannot open {camera.device!r}"
                    )
                capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, camera.width)
                capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, camera.height)
                capture.set(cv2_module.CAP_PROP_FPS, camera.fps)
                if hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
                    capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(3):
                self._capture_frames()
        except BaseException:
            self.close()
            raise

    def capture(self) -> tuple[CameraFrame, ...]:
        frames = self._capture_frames()
        images: list[CameraFrame] = []
        for camera, frame in zip(self._cameras, frames):
            encoded, jpeg = self._cv2.imencode(
                ".jpg",
                frame,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            if not encoded:
                raise CameraError(
                    f"camera {camera.name!r} frame could not be JPEG encoded"
                )
            images.append(
                CameraFrame(
                    name=camera.name,
                    mime_type="image/jpeg",
                    data=jpeg.tobytes(),
                )
            )
        return tuple(images)

    def _capture_frames(self) -> tuple[Any, ...]:
        for camera, capture in zip(self._cameras, self._captures):
            if not capture.grab():
                raise CameraError(f"camera {camera.name!r} failed to grab a frame")
        frames: list[Any] = []
        for camera, capture in zip(self._cameras, self._captures):
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise CameraError(f"camera {camera.name!r} failed to retrieve a frame")
            height, width = frame.shape[:2]
            if (width, height) != (camera.width, camera.height):
                raise CameraError(
                    f"camera {camera.name!r} returned {width}x{height}; "
                    f"expected {camera.width}x{camera.height}"
                )
            frames.append(frame)
        return tuple(frames)

    def close(self) -> None:
        for capture in self._captures:
            capture.release()
        self._captures.clear()


__all__ = ["CameraError", "V4L2CameraConfig", "V4L2CameraSource"]
