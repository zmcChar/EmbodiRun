"""In-memory directory for registered multi-tenant robot sessions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodied_runtime.distributed.session import RobotSessionIdentity

from .errors import SessionOverloadedError, SessionRegistrationError
from .types import SessionSnapshot, SessionState


class SessionDirectory:
    """Manage session membership and counters under the service's lock."""

    def __init__(self, *, max_sessions: int, idle_ttl_s: float) -> None:
        self.max_sessions = max_sessions
        self.idle_ttl_s = idle_ttl_s
        self._sessions: dict[str, SessionState] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def expire(self, now: float) -> None:
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if not state.pending_requests
            and (state.unregister_requested or now - state.last_seen_s > self.idle_ttl_s)
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def register(
        self,
        identity: RobotSessionIdentity,
        metadata: Mapping[str, Any],
        *,
        now: float,
    ) -> SessionSnapshot:
        registration_metadata = dict(metadata)
        existing = self._sessions.get(identity.session_id)
        if existing is not None:
            if existing.identity != identity:
                raise SessionRegistrationError(
                    f"session_id {identity.session_id!r} is already bound to "
                    f"robot {existing.identity.robot_id!r} and edge "
                    f"{existing.identity.edge_node_id!r}"
                )
            if existing.unregister_requested:
                raise SessionRegistrationError(f"session {identity.session_id!r} is unregistering")
            if existing.registration_metadata != registration_metadata:
                raise SessionRegistrationError(
                    f"session {identity.session_id!r} registration metadata "
                    "changed; rotate session_id for a new edge placement"
                )
            existing.last_seen_s = now
            return existing.snapshot()
        if len(self._sessions) >= self.max_sessions:
            raise SessionRegistrationError(f"session capacity reached ({self.max_sessions})")
        state = SessionState(
            identity=identity,
            registration_metadata=registration_metadata,
            last_seen_s=now,
        )
        self._sessions[identity.session_id] = state
        return state.snapshot()

    def unregister(self, identity: RobotSessionIdentity) -> None:
        state = self._sessions.get(identity.session_id)
        if state is None:
            return
        self._require_identity(state, identity)
        state.unregister_requested = True
        if not state.pending_requests:
            del self._sessions[identity.session_id]

    def snapshots(self) -> tuple[SessionSnapshot, ...]:
        return tuple(self._sessions[session_id].snapshot() for session_id in sorted(self._sessions))

    def begin_request(
        self,
        identity: RobotSessionIdentity,
        *,
        max_pending: int,
        now: float,
    ) -> SessionState:
        try:
            state = self._sessions[identity.session_id]
        except KeyError as error:
            raise SessionRegistrationError(
                f"session {identity.session_id!r} is not registered"
            ) from error
        self._require_identity(state, identity)
        if state.unregister_requested:
            raise SessionRegistrationError(f"session {identity.session_id!r} is unregistering")
        if state.pending_requests >= max_pending:
            raise SessionOverloadedError(
                f"session {identity.session_id!r} already has "
                f"{state.pending_requests} pending request(s)"
            )
        state.pending_requests += 1
        state.last_seen_s = now
        return state

    def finish_request(
        self,
        identity: RobotSessionIdentity,
        state: SessionState,
        *,
        now: float,
    ) -> None:
        state.pending_requests -= 1
        state.last_seen_s = now
        if (
            state.unregister_requested
            and not state.pending_requests
            and self._sessions.get(identity.session_id) is state
        ):
            del self._sessions[identity.session_id]

    @staticmethod
    def _require_identity(
        state: SessionState,
        identity: RobotSessionIdentity,
    ) -> None:
        if state.identity != identity:
            raise SessionRegistrationError(f"session identity mismatch for {identity.session_id!r}")


__all__ = ["SessionDirectory"]
