"""Session-bound multi-tenant TCP inference endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Mapping
from typing import Any

from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult
from embodied_runtime.engine.status import RequestStatus
from embodied_runtime.models.request import RawRequest

from ..tcp_json import TcpJsonProtocolError, TcpJsonRequestClient
from .errors import CloudSessionProtocolError, CloudSessionRemoteError
from .normalization import json_compatible, prepare_request_values
from .protocol import MULTI_ROBOT_PROTOCOL_VERSION
from .validation import (
    nonnegative_timing,
    validate_action_output,
    validate_base_response,
    validate_bound_metadata,
    validate_inference_response,
    validate_registration_response,
)


class MultiTenantTcpEndpoint:
    """Bind one edge runtime to one robot session on a shared cloud service."""

    def __init__(
        self,
        client: TcpJsonRequestClient,
        identity: RobotSessionIdentity,
        *,
        registration_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.identity = identity
        self.registration_metadata = dict(registration_metadata or {})
        self._connected = False
        self._enabled = True
        self._connection_epoch = 0
        self._connect_lock = asyncio.Lock()
        self._registered = False
        self._closed = False
        self._server_metadata: dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        return self._enabled and self._connected and not self._closed

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._closed

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    @property
    def server_metadata(self) -> Mapping[str, Any]:
        return dict(self._server_metadata)

    async def connect(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("cloud session endpoint is closed")
        if not self._enabled:
            raise ConnectionError("cloud session endpoint is logically disabled")

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("cloud session endpoint is closed")
            if not self._enabled:
                raise ConnectionError("cloud session endpoint is logically disabled")
            try:
                response = await self.client.request(
                    {
                        "kind": "register",
                        "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
                        **self.identity.to_wire(),
                        "metadata": json_compatible(
                            self.registration_metadata,
                            path="registration_metadata",
                        ),
                    }
                )
                validate_base_response(
                    response,
                    expected_kind="registered",
                    identity=self.identity,
                )
                validate_registration_response(
                    response,
                    identity=self.identity,
                    registration_metadata=self.registration_metadata,
                )
            except BaseException:
                self._transition_connected(False)
                raise
            self._server_metadata = dict(response)
            self._registered = True
            self._transition_connected(True)
            return dict(response)

    def set_enabled(self, enabled: bool) -> None:
        if self._closed and enabled:
            raise RuntimeError("cloud session endpoint is closed")
        self._enabled = enabled
        if not enabled:
            self._transition_connected(False)

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        if not self.connected:
            raise ConnectionError("cloud session endpoint is disconnected")
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        if not isinstance(request.payload, RawRequest):
            raise TypeError(
                "multi-tenant cloud requests must carry RawRequest so each "
                "robot's observation reaches the shared model"
            )
        validate_bound_metadata(request.metadata, self.identity)
        validate_bound_metadata(request.payload.metadata, self.identity)

        try:
            sequence_id = int(request.metadata["sequence_id"])
        except KeyError as error:
            raise ValueError("cloud request metadata requires sequence_id") from error
        observation_id = str(request.metadata.get("observation_id") or request.request_id)
        observation_timestamp_s = float(
            request.metadata.get(
                "observation_timestamp_s",
                request.created_at_s,
            )
        )
        raw = request.payload
        observation, request_metadata, raw_metadata = await asyncio.to_thread(
            prepare_request_values,
            raw,
            request.metadata,
        )
        response: Mapping[str, Any]
        try:
            response = await self.client.request(
                {
                    "kind": "infer",
                    "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
                    **self.identity.to_wire(),
                    "request_id": request.request_id,
                    "sequence_id": sequence_id,
                    "observation_id": observation_id,
                    "observation_timestamp_s": observation_timestamp_s,
                    "observation": observation,
                    "prompt": raw.prompt,
                    "request_metadata": request_metadata,
                    "raw_metadata": raw_metadata,
                    "deadline_s": request.deadline_s,
                    "num_steps": request.num_steps,
                    "seed": request.seed,
                }
            )
        except (
            ConnectionError,
            OSError,
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            TcpJsonProtocolError,
        ):
            self._transition_connected(False)
            raise

        if (
            response.get("ok") is not True
            and response.get("error_type") == "SessionRegistrationError"
        ):
            # A restarted service has forgotten this direct registration.
            # Changing the epoch invalidates cached authority and lets the edge
            # runtime register again in the background.
            self._registered = False
            self._transition_connected(False)
        try:
            validate_base_response(
                response,
                expected_kind="inference_result",
                identity=self.identity,
            )
            validate_inference_response(
                response,
                request_id=request.request_id,
                sequence_id=sequence_id,
                observation_id=observation_id,
            )
            metadata = response.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise CloudSessionProtocolError("cloud result metadata must be an object")
            status_value = response.get("status")
            if not isinstance(status_value, str):
                raise CloudSessionProtocolError("cloud result status must be a string")
            try:
                status = RequestStatus(status_value)
            except ValueError as error:
                raise CloudSessionProtocolError("cloud result has invalid status") from error
            queue_time_s = nonnegative_timing(response, "queue_time_s")
            execution_time_s = nonnegative_timing(response, "execution_time_s")
            output = response.get("output")
            if output is None:
                raise CloudSessionProtocolError("cloud result has no output")
            validate_action_output(output, self._server_metadata)
        except CloudSessionProtocolError:
            self._transition_connected(False)
            raise
        if status is not RequestStatus.SUCCEEDED:
            raise CloudSessionRemoteError(
                "CloudInferenceFailed",
                f"cloud inference completed with status {status.value!r}",
            )
        return InferenceResult(
            request_id=request.request_id,
            output=output,
            status=status,
            queue_time_s=queue_time_s,
            execution_time_s=execution_time_s,
            metadata={
                **metadata,
                **self.identity.to_wire(),
                "sequence_id": sequence_id,
                "observation_id": observation_id,
                "transport": "length_prefixed_json_v1",
            },
        )

    async def aclose(
        self,
        *,
        unregister_timeout_s: float | None = None,
    ) -> None:
        if unregister_timeout_s is not None and (
            not math.isfinite(unregister_timeout_s) or unregister_timeout_s <= 0
        ):
            raise ValueError("unregister_timeout_s must be finite and greater than zero")
        async with self._connect_lock:
            if self._closed:
                return
            self._enabled = False
            self._transition_connected(False)
            should_unregister = self._registered
            self._registered = False
            self._closed = True
            if should_unregister:
                with contextlib.suppress(Exception):
                    unregister = self.client.request(
                        {
                            "kind": "unregister",
                            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
                            **self.identity.to_wire(),
                        }
                    )
                    response = (
                        await unregister
                        if unregister_timeout_s is None
                        else await asyncio.wait_for(
                            unregister,
                            timeout=unregister_timeout_s,
                        )
                    )
                    validate_base_response(
                        response,
                        expected_kind="unregistered",
                        identity=self.identity,
                    )

    def _transition_connected(self, connected: bool) -> None:
        if connected != self._connected:
            self._connected = connected
            self._connection_epoch += 1
