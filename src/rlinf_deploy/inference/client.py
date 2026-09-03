"""Transport-independent policy client contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .contracts import PolicyObservation, PolicyResult, Session


class PolicyClient(Protocol):
    """Session-oriented client implemented by each inference transport."""

    def health(self) -> dict[str, Any]: ...

    def capabilities(self) -> dict[str, Any]: ...

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session: ...

    def step(self, observation: PolicyObservation) -> PolicyResult: ...

    def reset(self, session_id: str, *, request_id: str) -> Session: ...

    def close(self, session_id: str) -> None: ...


__all__ = ["PolicyClient"]
