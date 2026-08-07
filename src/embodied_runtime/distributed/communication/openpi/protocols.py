"""Injectable OpenPI codec and synchronous WebSocket protocols."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OpenPIWebSocketConnection(Protocol):
    """Minimal synchronous WebSocket surface used by the endpoint."""

    def send(self, message: bytes) -> Any: ...

    def recv(self, timeout: float | None = None) -> bytes | str: ...

    def close(self) -> Any: ...


@runtime_checkable
class OpenPIMessageCodec(Protocol):
    """Numpy-aware codec used for OpenPI frames."""

    def pack(self, value: Any) -> bytes: ...

    def unpack(self, value: bytes | str) -> Any: ...


OpenPIConnectionFactory = Callable[[str, float], OpenPIWebSocketConnection]
