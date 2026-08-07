"""Session isolation above one shared cloud inference provider."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterable, Mapping
from threading import RLock
from typing import Any

from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.engine.provider import InferenceProvider, ProviderCapabilities
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult

from .directory import SessionDirectory
from .errors import SessionRegistrationError, SessionSequenceError
from .normalization import (
    normalize_registration,
    prepare_provider_request,
    restore_external_result,
    validate_inference_arguments,
)
from .types import SessionSnapshot, SharedModelContract


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
        self._directory = SessionDirectory(
            max_sessions=max_sessions,
            idle_ttl_s=session_idle_ttl_s,
        )
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
            self._directory.expire(self._clock())
            return len(self._directory)

    def register(
        self,
        identity: RobotSessionIdentity,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionSnapshot:
        registration_metadata = normalize_registration(self.contract, identity, metadata)
        now = self._clock()
        with self._lock:
            if self._closed:
                raise RuntimeError("multi-tenant inference service is closed")
            self._directory.expire(now)
            return self._directory.register(
                identity,
                registration_metadata,
                now=now,
            )

    def unregister(self, identity: RobotSessionIdentity) -> None:
        with self._lock:
            self._directory.unregister(identity)

    def sessions(self) -> tuple[SessionSnapshot, ...]:
        with self._lock:
            self._directory.expire(self._clock())
            return self._directory.snapshots()

    async def infer_async(
        self,
        identity: RobotSessionIdentity,
        request: InferenceRequest,
        *,
        sequence_id: int,
        observation_id: str,
        observation_timestamp_s: float,
    ) -> InferenceResult:
        normalized_observation_id = validate_inference_arguments(
            request,
            sequence_id=sequence_id,
            observation_id=observation_id,
            observation_timestamp_s=observation_timestamp_s,
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("multi-tenant inference service is closed")
            self._directory.expire(self._clock())
            state = self._directory.begin_request(
                identity,
                max_pending=self.max_pending_per_session,
                now=self._clock(),
            )
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

                external_request_id, provider_request = prepare_provider_request(
                    identity,
                    request,
                    sequence_id=sequence_id,
                    observation_id=normalized_observation_id,
                    observation_timestamp_s=observation_timestamp_s,
                )
                state.last_sequence_id = sequence_id
                state.last_observation_id = normalized_observation_id
                state.accepted_requests += 1
                state.in_flight += 1
                try:
                    result = await self.provider.infer_async(provider_request)
                    if result.request_id != provider_request.request_id:
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

                return restore_external_result(
                    self.capabilities,
                    identity,
                    result,
                    external_request_id=external_request_id,
                    sequence_id=sequence_id,
                    observation_id=normalized_observation_id,
                    observation_timestamp_s=observation_timestamp_s,
                )
        finally:
            with self._lock:
                self._directory.finish_request(
                    identity,
                    state,
                    now=self._clock(),
                )
                self._active_calls -= 1
                if not self._active_calls:
                    self._drained.set()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._provider_closed:
                return
            with self._lock:
                self._closed = True
            await self._drained.wait()
            await self.provider.aclose()
            self._provider_closed = True


__all__ = ["MultiTenantInferenceService"]
