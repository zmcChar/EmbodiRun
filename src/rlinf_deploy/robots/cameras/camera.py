"""Model-independent camera values used by robot runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One encoded frame identified in the robot observation schema."""

    name: str
    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera frame name must not be empty")
        if self.mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError("camera frame must be JPEG or PNG")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("camera frame data must not be empty")


class CameraSource(Protocol):
    """Capture and release a configured collection of cameras."""

    def capture(self) -> tuple[CameraFrame, ...]: ...

    def close(self) -> None: ...


__all__ = ["CameraFrame", "CameraSource"]
