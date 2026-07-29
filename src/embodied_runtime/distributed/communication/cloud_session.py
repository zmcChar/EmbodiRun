"""Robot-session-aware endpoint over the prototype TCP/JSON transport."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    RawRequest,
    RequestStatus,
)
from embodied_runtime.distributed.session import RobotSessionIdentity

from .tcp_json import TcpJsonProtocolError, TcpJsonRequestClient

MULTI_ROBOT_PROTOCOL_VERSION = 1


class CloudSessionProtocolError(RuntimeError):
    """The shared cloud service returned an invalid session response."""


class CloudSessionRemoteError(RuntimeError):
    """The shared cloud service rejected a well-formed request."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.remote_message = message
        super().__init__(f"{error_type}: {message}")


def json_compatible(value: Any, *, path: str = "value") -> Any:
    """Convert tensor-like trees to strict JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return json_compatible(value.value, path=path)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            converted[key] = json_compatible(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [json_compatible(item, path=f"{path}[{index}]") for index, item in enumerate(value)]

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        return json_compatible(tolist(), path=path)
    item = getattr(candidate, "item", None)
    if callable(item):
        return json_compatible(item(), path=path)
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def _response_error(response: Mapping[str, Any]) -> CloudSessionRemoteError:
    error_type = str(response.get("error_type") or "CloudError")
    message = str(response.get("error") or "cloud request failed")
    return CloudSessionRemoteError(error_type, message)


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
                self._validate_base_response(response, expected_kind="registered")
                self._validate_registration_response(response)
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
        self._validate_bound_metadata(request.metadata)
        self._validate_bound_metadata(request.payload.metadata)

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
            self._prepare_request_values,
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
            self._validate_base_response(
                response,
                expected_kind="inference_result",
            )
            self._validate_inference_response(
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
            queue_time_s = self._nonnegative_timing(response, "queue_time_s")
            execution_time_s = self._nonnegative_timing(
                response,
                "execution_time_s",
            )
            output = response.get("output")
            if output is None:
                raise CloudSessionProtocolError("cloud result has no output")
            self._validate_action_output(output)
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
                    self._validate_base_response(
                        response,
                        expected_kind="unregistered",
                    )

    def _validate_bound_metadata(self, metadata: Mapping[str, Any]) -> None:
        for name, expected in self.identity.to_wire().items():
            supplied = metadata.get(name)
            if supplied is not None and str(supplied) != expected:
                raise ValueError(
                    f"request metadata cannot override bound {name}: {supplied!r} != {expected!r}"
                )

    @staticmethod
    def _prepare_request_values(
        raw: RawRequest,
        request_metadata: Mapping[str, Any],
    ) -> tuple[Any, Any, Any]:
        return (
            json_compatible(raw.observation, path="observation"),
            json_compatible(request_metadata, path="request_metadata"),
            json_compatible(raw.metadata, path="raw_metadata"),
        )

    @staticmethod
    def _nonnegative_timing(
        response: Mapping[str, Any],
        name: str,
    ) -> float:
        raw = response.get(name)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise CloudSessionProtocolError(f"cloud result {name} must be a number")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise CloudSessionProtocolError(f"cloud result {name} must be finite and non-negative")
        return value

    def _validate_registration_response(
        self,
        response: Mapping[str, Any],
    ) -> None:
        supported = response.get("supported_embodiments")
        if (
            not isinstance(supported, list)
            or any(not isinstance(item, str) for item in supported)
            or self.identity.embodiment not in supported
        ):
            raise CloudSessionProtocolError("cloud registration has invalid supported_embodiments")

        for name, edge_name in (
            ("action_dim", "edge_action_dim"),
            ("action_horizon", "edge_action_horizon"),
        ):
            value = response.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CloudSessionProtocolError(
                    f"cloud registration {name} must be a positive integer"
                )
            edge_value = self.registration_metadata.get(edge_name)
            if edge_value is not None and edge_value != value:
                raise CloudSessionProtocolError(
                    f"cloud registration {name} does not match the edge contract"
                )

        last_sequence_id = response.get("last_sequence_id")
        if (
            not isinstance(last_sequence_id, int)
            or isinstance(last_sequence_id, bool)
            or last_sequence_id < 0
        ):
            raise CloudSessionProtocolError(
                "cloud registration last_sequence_id must be a non-negative integer"
            )

    def _validate_action_output(self, output: Any) -> None:
        """Reject malformed cloud actions before they can become authoritative."""

        if not isinstance(output, Mapping):
            raise CloudSessionProtocolError("cloud output must be an object")
        actions = output.get("actions")
        if not isinstance(actions, list) or not actions:
            raise CloudSessionProtocolError("cloud output.actions must be a non-empty array")

        action_dim = self._server_metadata.get("action_dim")
        action_horizon = self._server_metadata.get("action_horizon")
        if action_dim is None or action_horizon is None:
            raise CloudSessionProtocolError("cloud registration did not declare an action shape")
        expected_dim = int(action_dim)
        expected_horizon = int(action_horizon)

        if all(self._is_finite_number(value) for value in actions):
            rows = [actions]
        elif all(isinstance(row, list) for row in actions):
            rows = actions
        else:
            raise CloudSessionProtocolError(
                "cloud output.actions must be a numeric vector or matrix"
            )

        if len(rows) != expected_horizon:
            raise CloudSessionProtocolError(
                "cloud output action horizon does not match the registered contract"
            )
        for row in rows:
            if len(row) != expected_dim or not all(self._is_finite_number(value) for value in row):
                raise CloudSessionProtocolError(
                    "cloud output action dimension or values do not match the registered contract"
                )

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def _validate_base_response(
        self,
        response: Mapping[str, Any],
        *,
        expected_kind: str,
    ) -> None:
        if response.get("ok") is not True:
            raise _response_error(response)
        if response.get("protocol_version") != MULTI_ROBOT_PROTOCOL_VERSION:
            raise CloudSessionProtocolError("cloud protocol version mismatch")
        if response.get("kind") != expected_kind:
            raise CloudSessionProtocolError(
                f"expected cloud response kind {expected_kind!r}, got {response.get('kind')!r}"
            )
        for name, expected in self.identity.to_wire().items():
            if response.get(name) != expected:
                raise CloudSessionProtocolError(
                    f"cloud response {name} does not match the bound session"
                )

    @staticmethod
    def _validate_inference_response(
        response: Mapping[str, Any],
        *,
        request_id: str,
        sequence_id: int,
        observation_id: str,
    ) -> None:
        expected = {
            "request_id": request_id,
            "sequence_id": sequence_id,
            "observation_id": observation_id,
        }
        for name, value in expected.items():
            if response.get(name) != value:
                raise CloudSessionProtocolError(f"cloud response {name} does not match the request")

    def _transition_connected(self, connected: bool) -> None:
        if connected != self._connected:
            self._connected = connected
            self._connection_epoch += 1


__all__ = [
    "MULTI_ROBOT_PROTOCOL_VERSION",
    "CloudSessionProtocolError",
    "CloudSessionRemoteError",
    "MultiTenantTcpEndpoint",
    "json_compatible",
]
