"""Session isolation above one shared cloud inference provider."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any

from embodied_runtime.contracts import InferenceRequest, InferenceResult, RawRequest
from embodied_runtime.distributed.session import RobotSessionIdentity

from .base import InferenceProvider, ProviderCapabilities


class SessionRegistrationError(RuntimeError):
    """A robot session is missing or conflicts with an existing registration."""


class SessionSequenceError(RuntimeError):
    """A request is duplicated or arrives out of order within one session."""


class SessionOverloadedError(RuntimeError):
    """A session already has the maximum allowed pending requests."""


@dataclass(frozen=True, slots=True)
class SharedModelContract:
    """Action and embodiment contract of one shared cloud model."""

    action_space_id: str
    supported_embodiments: frozenset[str]
    action_dim: int | None
    action_horizon: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.action_space_id, str):
            raise TypeError("action_space_id must be a string")
        normalized_action_space = self.action_space_id.strip()
        if not normalized_action_space:
            raise ValueError("action_space_id must not be empty")
        if not self.supported_embodiments:
            raise ValueError("supported_embodiments must not be empty")
        if any(
            not isinstance(embodiment, str) or not embodiment.strip()
            for embodiment in self.supported_embodiments
        ):
            raise ValueError("supported embodiment names must not be empty")
        normalized_embodiments = frozenset(
            embodiment.strip() for embodiment in self.supported_embodiments
        )
        object.__setattr__(self, "action_space_id", normalized_action_space)
        object.__setattr__(
            self,
            "supported_embodiments",
            normalized_embodiments,
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
class _SessionState:
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


class MultiTenantInferenceService:
    """Serve independent robot sessions through one loaded Provider.

    Requests are serialized within one session and may execute concurrently
    across sessions. A local Provider's ExecutionEngine can therefore batch
    compatible observations from different robots while preserving per-robot
    sequence order.
    """

    def __init__(
        self,
        provider: InferenceProvider,
        *,
        action_space_id: str,
        supported_embodiments: Iterable[str],
        max_sessions: int = 128,
        max_pending_per_session: int = 1,
        session_idle_ttl_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(provider, InferenceProvider):
            raise TypeError("provider does not satisfy InferenceProvider")
        if "multi_tenant_safe" not in provider.capabilities.features:
            raise ValueError(
                "the shared-provider service requires an explicit "
                "'multi_tenant_safe' Provider capability; stateful Providers "
                "need one session-bound instance per robot"
            )
        if max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")
        if max_pending_per_session <= 0:
            raise ValueError("max_pending_per_session must be greater than zero")
        if not math.isfinite(session_idle_ttl_s) or session_idle_ttl_s <= 0:
            raise ValueError("session_idle_ttl_s must be finite and greater than zero")
        if isinstance(supported_embodiments, (str, bytes)):
            raise TypeError("supported_embodiments must be an iterable of names")
        self.provider = provider
        model = provider.capabilities.model
        self.contract = SharedModelContract(
            action_space_id=action_space_id,
            supported_embodiments=frozenset(supported_embodiments),
            action_dim=model.action_dim,
            action_horizon=model.action_horizon,
        )
        self.max_sessions = max_sessions
        self.max_pending_per_session = max_pending_per_session
        self.session_idle_ttl_s = session_idle_ttl_s
        self._clock = clock
        self._sessions: dict[str, _SessionState] = {}
        self._lock = RLock()
        self._active_calls = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._provider_closed = False

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.provider.capabilities

    @property
    def registered_session_count(self) -> int:
        with self._lock:
            self._expire_idle_sessions(self._clock())
            return len(self._sessions)

    def register(
        self,
        identity: RobotSessionIdentity,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionSnapshot:
        if identity.action_space_id != self.contract.action_space_id:
            raise SessionRegistrationError(
                f"action_space_id {identity.action_space_id!r} is incompatible "
                f"with cloud contract {self.contract.action_space_id!r}"
            )
        if identity.embodiment not in self.contract.supported_embodiments:
            supported = ", ".join(sorted(self.contract.supported_embodiments))
            raise SessionRegistrationError(
                f"embodiment {identity.embodiment!r} is unsupported; cloud supports: {supported}"
            )
        registration_metadata = dict(metadata or {})
        self._validate_edge_action_shape(registration_metadata)
        now = self._clock()
        with self._lock:
            if self._closed:
                raise RuntimeError("multi-tenant inference service is closed")
            self._expire_idle_sessions(now)
            existing = self._sessions.get(identity.session_id)
            if existing is not None:
                if existing.identity != identity:
                    raise SessionRegistrationError(
                        f"session_id {identity.session_id!r} is already bound to "
                        f"robot {existing.identity.robot_id!r} and edge "
                        f"{existing.identity.edge_node_id!r}"
                    )
                if existing.unregister_requested:
                    raise SessionRegistrationError(
                        f"session {identity.session_id!r} is unregistering"
                    )
                if existing.registration_metadata != registration_metadata:
                    raise SessionRegistrationError(
                        f"session {identity.session_id!r} registration metadata "
                        "changed; rotate session_id for a new edge placement"
                    )
                existing.last_seen_s = now
                return existing.snapshot()
            if len(self._sessions) >= self.max_sessions:
                raise SessionRegistrationError(f"session capacity reached ({self.max_sessions})")
            state = _SessionState(
                identity=identity,
                registration_metadata=registration_metadata,
                last_seen_s=now,
            )
            self._sessions[identity.session_id] = state
            return state.snapshot()

    def unregister(self, identity: RobotSessionIdentity) -> None:
        with self._lock:
            state = self._sessions.get(identity.session_id)
            if state is None:
                return
            self._require_identity(state, identity)
            state.unregister_requested = True
            if not state.pending_requests:
                del self._sessions[identity.session_id]

    def sessions(self) -> tuple[SessionSnapshot, ...]:
        with self._lock:
            self._expire_idle_sessions(self._clock())
            return tuple(
                self._sessions[session_id].snapshot() for session_id in sorted(self._sessions)
            )

    async def infer_async(
        self,
        identity: RobotSessionIdentity,
        request: InferenceRequest,
        *,
        sequence_id: int,
        observation_id: str,
        observation_timestamp_s: float,
    ) -> InferenceResult:
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        if sequence_id <= 0:
            raise ValueError("sequence_id must be greater than zero")
        normalized_observation_id = observation_id.strip()
        if not normalized_observation_id:
            raise ValueError("observation_id must not be empty")
        if not math.isfinite(observation_timestamp_s) or observation_timestamp_s < 0:
            raise ValueError("observation_timestamp_s must be finite and non-negative")

        with self._lock:
            if self._closed:
                raise RuntimeError("multi-tenant inference service is closed")
            self._expire_idle_sessions(self._clock())
            try:
                state = self._sessions[identity.session_id]
            except KeyError as error:
                raise SessionRegistrationError(
                    f"session {identity.session_id!r} is not registered"
                ) from error
            self._require_identity(state, identity)
            if state.unregister_requested:
                raise SessionRegistrationError(f"session {identity.session_id!r} is unregistering")
            if state.pending_requests >= self.max_pending_per_session:
                raise SessionOverloadedError(
                    f"session {identity.session_id!r} already has "
                    f"{state.pending_requests} pending request(s)"
                )
            state.pending_requests += 1
            state.last_seen_s = self._clock()
            self._active_calls += 1
            self._drained.clear()

        try:
            async with state.lock:
                if state.unregister_requested:
                    raise SessionRegistrationError(
                        f"session {identity.session_id!r} is unregistering"
                    )
                if sequence_id <= state.last_sequence_id:
                    raise SessionSequenceError(
                        f"session {identity.session_id!r} received sequence_id "
                        f"{sequence_id}, expected greater than "
                        f"{state.last_sequence_id}"
                    )

                external_request_id = request.request_id
                internal_request_id = identity.namespace_request_id(external_request_id)
                authoritative_metadata = {
                    **request.metadata,
                    **identity.to_wire(),
                    "sequence_id": sequence_id,
                    "observation_id": normalized_observation_id,
                    "observation_timestamp_s": observation_timestamp_s,
                    "external_request_id": external_request_id,
                }
                payload = request.payload
                if isinstance(payload, RawRequest):
                    payload = replace(
                        payload,
                        metadata={
                            **payload.metadata,
                            **identity.to_wire(),
                            "sequence_id": sequence_id,
                            "observation_id": normalized_observation_id,
                            "observation_timestamp_s": observation_timestamp_s,
                        },
                    )
                provider_request = replace(
                    request,
                    payload=payload,
                    request_id=internal_request_id,
                    # Remote priority is intentionally not trusted by the
                    # shared service in this first slice.
                    priority=0,
                    metadata=authoritative_metadata,
                )
                state.last_sequence_id = sequence_id
                state.last_observation_id = normalized_observation_id
                state.accepted_requests += 1
                state.in_flight += 1
                try:
                    result = await self.provider.infer_async(provider_request)
                    if result.request_id != internal_request_id:
                        raise RuntimeError(
                            "shared Provider returned a request_id from another request"
                        )
                except BaseException:
                    state.failed_requests += 1
                    raise
                else:
                    state.completed_requests += 1
                finally:
                    state.in_flight -= 1

                capabilities = self.capabilities
                device = capabilities.device
                return InferenceResult(
                    request_id=external_request_id,
                    output=result.output,
                    status=result.status,
                    queue_time_s=result.queue_time_s,
                    execution_time_s=result.execution_time_s,
                    metadata={
                        **result.metadata,
                        "model_id": capabilities.model.model_id,
                        "provider": capabilities.name,
                        "provider_runtime": capabilities.runtime,
                        "backend": (device.backend if device is not None else None),
                        "device_id": (device.device_id if device is not None else None),
                        **identity.to_wire(),
                        "sequence_id": sequence_id,
                        "observation_id": normalized_observation_id,
                        "observation_timestamp_s": observation_timestamp_s,
                        "provider_request_id": result.request_id,
                    },
                )
        finally:
            with self._lock:
                state.pending_requests -= 1
                state.last_seen_s = self._clock()
                self._active_calls -= 1
                if not self._active_calls:
                    self._drained.set()
                if (
                    state.unregister_requested
                    and not state.pending_requests
                    and self._sessions.get(identity.session_id) is state
                ):
                    del self._sessions[identity.session_id]

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._provider_closed:
                return
            with self._lock:
                self._closed = True
            await self._drained.wait()
            await self.provider.aclose()
            self._provider_closed = True

    @staticmethod
    def _require_identity(
        state: _SessionState,
        identity: RobotSessionIdentity,
    ) -> None:
        if state.identity != identity:
            raise SessionRegistrationError(f"session identity mismatch for {identity.session_id!r}")

    def _validate_edge_action_shape(
        self,
        metadata: Mapping[str, Any],
    ) -> None:
        expected = {
            "edge_action_dim": self.contract.action_dim,
            "edge_action_horizon": self.contract.action_horizon,
        }
        for name, value in expected.items():
            if value is None:
                continue
            supplied = metadata.get(name)
            if supplied is None:
                raise SessionRegistrationError(f"registration metadata requires {name}")
            if not isinstance(supplied, int) or isinstance(supplied, bool) or supplied != value:
                raise SessionRegistrationError(
                    f"{name} {supplied!r} is incompatible with cloud value {value}"
                )

    def _expire_idle_sessions(self, now: float) -> None:
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if not state.pending_requests
            and (state.unregister_requested or now - state.last_seen_s > self.session_idle_ttl_s)
        ]
        for session_id in expired:
            del self._sessions[session_id]


__all__ = [
    "MultiTenantInferenceService",
    "SessionOverloadedError",
    "SessionRegistrationError",
    "SessionSequenceError",
    "SessionSnapshot",
    "SharedModelContract",
]
