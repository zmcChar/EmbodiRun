"""Robot-independent camera capture contracts."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from typing import Protocol

from rlinf_deploy.inference import ImagePayload


class CameraSource(Protocol):
    """One named camera stream that produces encoded policy images."""

    name: str

    def connect(self) -> None: ...

    def capture(self) -> ImagePayload: ...

    def close(self) -> None: ...


class CameraRig:
    """Own the lifecycle of the cameras used for one policy observation."""

    def __init__(self, sources: Sequence[CameraSource]) -> None:
        sources = tuple(sources)
        if not sources:
            raise ValueError("camera rig must contain at least one source")
        names = tuple(source.name for source in sources)
        if len(set(names)) != len(names):
            raise ValueError("camera source names must be unique")
        self.sources = sources
        self._connections: ExitStack | None = None

    def connect(self) -> None:
        if self._connections is not None:
            raise RuntimeError("camera rig is already connected")
        connections = ExitStack()
        try:
            for source in self.sources:
                source.connect()
                connections.callback(source.close)
        except BaseException:
            connections.close()
            raise
        self._connections = connections

    def capture(self) -> tuple[ImagePayload, ...]:
        if self._connections is None:
            raise RuntimeError("camera rig is not connected")
        return tuple(source.capture() for source in self.sources)

    def close(self) -> None:
        connections, self._connections = self._connections, None
        if connections is not None:
            connections.close()

    def __enter__(self) -> CameraRig:  # noqa: PYI034 - Python 3.10 has no typing.Self
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["CameraRig", "CameraSource"]
