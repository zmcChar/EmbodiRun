from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..discovery import NodeRole, RuntimeEndpoint


@runtime_checkable
class RuntimeRegistry(Protocol):
    def register(self, endpoint: RuntimeEndpoint, lease_s: float) -> None: ...

    def heartbeat(self, endpoint_id: str, lease_s: float) -> None: ...

    def unregister(self, endpoint_id: str) -> None: ...

    def list(self, role: NodeRole | None = None) -> Sequence[RuntimeEndpoint]: ...
