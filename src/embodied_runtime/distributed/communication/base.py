from __future__ import annotations

from typing import Protocol, runtime_checkable

from embodied_runtime.contracts import Envelope


@runtime_checkable
class Transport(Protocol):
    def send(self, envelope: Envelope) -> None: ...

    def receive(self, timeout_s: float | None = None) -> Envelope | None: ...

    def close(self) -> None: ...
