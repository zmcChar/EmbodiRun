"""Robot-scoped edge/cloud failover runtime and reconnect lifecycle."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FailoverDecision,
    ResultSource,
)
from embodied_runtime.distributed.communication import (
    CloudSessionProtocolError,
    CloudSessionRemoteError,
    MultiTenantTcpEndpoint,
    TcpJsonProtocolError,
)
from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.engine.provider import InferenceProvider
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult

from .edge_codec import (
    prepare_edge_request,
    snapshot_observation,
    validate_cloud_registration,
    validate_cloud_result,
)


class RobotEdgeRuntime:
    """One edge Provider, one cloud session, and one failover state machine."""

    def __init__(
        self,
        *,
        identity: RobotSessionIdentity,
        edge: InferenceProvider,
        cloud: MultiTenantTcpEndpoint,
        failover: FailoverConfig,
        registration_timeout_s: float = 2.0,
        reconnect_interval_s: float = 1.0,
    ) -> None:
        if cloud.identity != identity:
            raise ValueError("cloud endpoint identity must match edge runtime")
        if not math.isfinite(reconnect_interval_s) or reconnect_interval_s < 0:
            raise ValueError("reconnect_interval_s must be finite and non-negative")
        if not math.isfinite(registration_timeout_s) or registration_timeout_s <= 0:
            raise ValueError("registration_timeout_s must be finite and greater than zero")
        self.identity = identity
        self.edge = edge
        self.cloud = cloud
        self.failover = AsyncFailoverCoordinator(edge=edge, cloud=cloud, config=failover)
        self.registration_timeout_s = registration_timeout_s
        self.reconnect_interval_s = reconnect_interval_s
        self._reconnect_task: asyncio.Task[Mapping[str, Any]] | None = None
        self._last_reconnect_attempt_s: float | None = None
        self._last_reconnect_error: BaseException | None = None
        self._last_sequence_id = 0
        self._inference_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._resources_closed = False

    async def start(self) -> None:
        """Attempt initial registration without making cloud availability mandatory."""

        if self._closed:
            raise RuntimeError("edge runtime is closed")
        try:
            await self._connect_with_timeout()
        except (
            CloudSessionProtocolError,
            CloudSessionRemoteError,
            ConnectionError,
            OSError,
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            TcpJsonProtocolError,
            ValueError,
        ) as error:
            self._last_reconnect_error = error
            return
        self._last_reconnect_error = None

    @property
    def last_reconnect_error(self) -> BaseException | None:
        return self._last_reconnect_error

    def set_cloud_enabled(self, enabled: bool) -> None:
        self.cloud.set_enabled(enabled)
        if enabled:
            self._last_reconnect_attempt_s = None
            self._schedule_reconnect()

    async def infer_observation(
        self,
        observation: Mapping[str, Any],
        *,
        sequence_id: int,
        observation_timestamp_s: float | None = None,
        prompt: str | None = None,
        owned_cloud_observation: Mapping[str, Any] | None = None,
    ) -> FailoverDecision[InferenceResult]:
        """Run one ordered control tick with at-most-once sequence admission."""

        async with self._inference_lock:
            if self._closed:
                raise RuntimeError("edge runtime is closed")
            expected_sequence_id = self._last_sequence_id + 1
            if sequence_id != expected_sequence_id:
                raise ValueError(
                    "sequence_id must be contiguous starting at 1; "
                    f"expected {expected_sequence_id}, received {sequence_id}"
                )
            self._last_sequence_id = sequence_id
            self._schedule_reconnect()
            timestamp_s = (
                time.monotonic() if observation_timestamp_s is None else observation_timestamp_s
            )
            prepared = prepare_edge_request(
                self.identity,
                observation,
                sequence_id=sequence_id,
                observation_timestamp_s=timestamp_s,
                prompt=prompt,
            )

            def build_cloud_request() -> InferenceRequest:
                cloud_observation = (
                    owned_cloud_observation
                    if owned_cloud_observation is not None
                    else snapshot_observation(observation)
                )
                return prepared.cloud_request(
                    cloud_observation,
                    deadline_s=self.failover.config.cloud_request_timeout_s,
                )

            decision = await self.failover.infer_async(
                prepared.request,
                cloud_request_factory=build_cloud_request,
            )
            if decision.source is ResultSource.EDGE:
                return replace(
                    decision,
                    result=replace(
                        decision.result,
                        metadata={**decision.result.metadata, **prepared.metadata},
                    ),
                )
            validate_cloud_result(self.identity, decision.result, sequence_id=sequence_id)
            return decision

    async def wait_for_cloud_idle(self) -> None:
        await self.failover.wait_for_cloud_idle()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._resources_closed:
                return
            self._closed = True
            async with self._inference_lock:
                pass
            reconnect = self._reconnect_task
            if reconnect is not None and not reconnect.done():
                reconnect.cancel()
                await asyncio.gather(reconnect, return_exceptions=True)
            await self.failover.aclose()
            await self.cloud.aclose(unregister_timeout_s=self.registration_timeout_s)
            await self.edge.aclose()
            self._resources_closed = True

    def _schedule_reconnect(self) -> None:
        if self._closed or self.cloud.connected or not self.cloud.enabled:
            return
        active = self._reconnect_task
        if active is not None and not active.done():
            return
        now = asyncio.get_running_loop().time()
        if (
            self._last_reconnect_attempt_s is not None
            and now - self._last_reconnect_attempt_s < self.reconnect_interval_s
        ):
            return
        self._last_reconnect_attempt_s = now
        task = asyncio.create_task(
            self._connect_with_timeout(),
            name=f"cloud-register-{self.identity.session_id}",
        )
        self._reconnect_task = task

        def consume_result(completed: asyncio.Task[Mapping[str, Any]]) -> None:
            if self._reconnect_task is completed:
                self._reconnect_task = None
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception as error:  # noqa: BLE001 - retain diagnostic failure
                self._last_reconnect_error = error
            else:
                self._last_reconnect_error = None

        task.add_done_callback(consume_result)

    async def _connect_with_timeout(self) -> Mapping[str, Any]:
        return await asyncio.wait_for(
            self._connect_and_validate(),
            timeout=self.registration_timeout_s,
        )

    async def _connect_and_validate(self) -> Mapping[str, Any]:
        try:
            response = await self.cloud.connect()
            validate_cloud_registration(
                self.identity,
                self.edge.capabilities,
                response,
                last_sequence_id=self._last_sequence_id,
            )
        except CloudSessionRemoteError as error:
            if error.error_type == "SessionRegistrationError":
                self.cloud.set_enabled(False)
            raise
        except (CloudSessionProtocolError, ValueError):
            self.cloud.set_enabled(False)
            raise
        return response


__all__ = ["RobotEdgeRuntime"]
