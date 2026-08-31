"""RGB camera devices adapted to encoded inference images."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any, Protocol

from rlinf_deploy.inference import ImagePayload


class RGBFrameDevice(Protocol):
    """Camera SDK boundary used by the generic encoded-image source."""

    def connect(self) -> None: ...

    def read(self, timeout_ms: float) -> Any: ...

    def close(self) -> None: ...


FrameEncoder = Callable[[Any, int], bytes]


def _encode_jpeg(frame: Any, quality: int) -> bytes:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - installed by camera extras
        raise RuntimeError(
            "camera encoding requires a camera backend such as "
            "rlinf-deploy[opencv]"
        ) from error

    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) != 3 or shape[2] != 3:
        raise ValueError("camera frame must have shape (height, width, 3)")
    image = Image.fromarray(frame)
    if image.mode != "RGB":
        raise ValueError(f"camera frame must be RGB, got {image.mode}")
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


class RGBCameraSource:
    """Encode frames from any RGB camera device for policy inference."""

    def __init__(
        self,
        name: str,
        device: RGBFrameDevice,
        *,
        timeout_ms: float = 1000,
        jpeg_quality: int = 90,
        encoder: FrameEncoder = _encode_jpeg,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.name = name
        self._device = device
        self._timeout_ms = timeout_ms
        self._jpeg_quality = jpeg_quality
        self._encoder = encoder
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError(f"camera {self.name!r} is already connected")
        try:
            self._device.connect()
        except BaseException:
            self._device.close()
            raise
        self._connected = True

    def capture(self) -> ImagePayload:
        if not self._connected:
            raise RuntimeError(f"camera {self.name!r} is not connected")
        frame = self._device.read(self._timeout_ms)
        return ImagePayload(
            name=self.name,
            mime_type="image/jpeg",
            data=self._encoder(frame, self._jpeg_quality),
        )

    def close(self) -> None:
        if not self._connected:
            return
        try:
            self._device.close()
        finally:
            self._connected = False


__all__ = ["RGBCameraSource", "RGBFrameDevice"]
