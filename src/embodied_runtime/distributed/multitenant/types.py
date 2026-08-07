"""Validated configuration and mutable state for shared model sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from embodied_runtime.distributed.session import RobotSessionIdentity


@dataclass(frozen=True, slots=True)
class SharedModelContract:
    action_space_id: str
    supported_embodiments: frozenset[str]
    action_dim: int | None
    action_horizon: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.action_space_id, str):
            raise TypeError("action_space_id must be a string")
        action_space = self.action_space_id.strip()
        if not action_space:
            raise ValueError("action_space_id must not be empty")
        if not self.supported_embodiments:
            raise ValueError("supported_embodiments must not be empty")
        if any(
            not isinstance(embodiment, str) or not embodiment.strip()
            for embodiment in self.supported_embodiments
        ):
            raise ValueError("supported embodiment names must not be empty")
        object.__setattr__(self, "action_space_id", action_space)
        object.__setattr__(
            self,
            "supported_embodiments",
            frozenset(item.strip() for item in self.supported_embodiments),
        )
        if self.action_dim is not None and self.action_dim <= 0:
            raise ValueError("action_dim must be greater than zero")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be greater than zero")


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    identity: RobotSessionIdentity
    registration_metadata: Mapping[str, Any]
    accepted_requests: int
    completed_requests: int
    failed_requests: int
    pending_requests: int
    in_flight: int
    last_sequence_id: int
    last_observation_id: str | None
    last_seen_s: float


@dataclass(slots=True)
class SessionState:
    identity: RobotSessionIdentity
    registration_metadata: dict[str, Any]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    accepted_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    pending_requests: int = 0
    in_flight: int = 0
    last_sequence_id: int = 0
    last_observation_id: str | None = None
    unregister_requested: bool = False
    last_seen_s: float = 0.0

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            identity=self.identity,
            registration_metadata=dict(self.registration_metadata),
            accepted_requests=self.accepted_requests,
            completed_requests=self.completed_requests,
            failed_requests=self.failed_requests,
            pending_requests=self.pending_requests,
            in_flight=self.in_flight,
            last_sequence_id=self.last_sequence_id,
            last_observation_id=self.last_observation_id,
            last_seen_s=self.last_seen_s,
        )


__all__ = ["SessionSnapshot", "SessionState", "SharedModelContract"]
