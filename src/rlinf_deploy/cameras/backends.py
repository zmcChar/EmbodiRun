"""Optional camera SDK backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .rgb import RGBCameraSource


class _LeRobotCamera(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect(self, warmup: bool = True) -> None: ...

    def async_read(self, timeout_ms: float = 200) -> Any: ...

    def disconnect(self) -> None: ...


class _LeRobotRGBDevice:
    """Keep the optional LeRobot camera SDK behind the generic device contract."""

    def __init__(self, camera: _LeRobotCamera) -> None:
        self._camera = camera

    def connect(self) -> None:
        self._camera.connect()

    def read(self, timeout_ms: float) -> Any:
        return self._camera.async_read(timeout_ms=timeout_ms)

    def close(self) -> None:
        if self._camera.is_connected:
            self._camera.disconnect()


def opencv_camera(
    name: str,
    index_or_path: int | Path,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    warmup_s: int = 1,
    timeout_ms: float = 1000,
    jpeg_quality: int = 90,
) -> RGBCameraSource:
    """Create an RGB source for a USB, laptop, or other OpenCV camera."""

    try:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    except ImportError as error:
        raise RuntimeError(
            "OpenCV camera support requires rlinf-deploy[opencv]"
        ) from error

    config = OpenCVCameraConfig(
        index_or_path=index_or_path,
        fps=fps,
        width=width,
        height=height,
        color_mode="rgb",
        warmup_s=warmup_s,
    )
    return RGBCameraSource(
        name,
        _LeRobotRGBDevice(OpenCVCamera(config)),
        timeout_ms=timeout_ms,
        jpeg_quality=jpeg_quality,
    )


def realsense_camera(
    name: str,
    serial_number_or_name: str,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    warmup_s: int = 1,
    timeout_ms: float = 1000,
    jpeg_quality: int = 90,
) -> RGBCameraSource:
    """Create an RGB-only source for an Intel RealSense camera."""

    try:
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
    except ImportError as error:
        raise RuntimeError(
            "RealSense camera support requires rlinf-deploy[realsense]"
        ) from error

    config = RealSenseCameraConfig(
        serial_number_or_name=serial_number_or_name,
        fps=fps,
        width=width,
        height=height,
        color_mode="rgb",
        use_rgb=True,
        use_depth=False,
        warmup_s=warmup_s,
    )
    return RGBCameraSource(
        name,
        _LeRobotRGBDevice(RealSenseCamera(config)),
        timeout_ms=timeout_ms,
        jpeg_quality=jpeg_quality,
    )


__all__ = ["opencv_camera", "realsense_camera"]
