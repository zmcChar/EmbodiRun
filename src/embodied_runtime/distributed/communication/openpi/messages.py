"""OpenPI request construction and response contract decoding."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.models.request import RawRequest

from .actions import validate_openpi_actions
from .errors import OpenPIProtocolError, OpenPIRemoteError, OpenPITimeoutError


def request_deadline(request: InferenceRequest, timeout_s: float) -> float:
    now = time.monotonic()
    endpoint_deadline = now + timeout_s
    if request.deadline_s is None:
        return endpoint_deadline
    if request.deadline_s <= 0 or not math.isfinite(request.deadline_s):
        raise ValueError("request deadline_s must be a finite value greater than zero")
    request_deadline_s = request.created_at_s + request.deadline_s
    if request_deadline_s <= now:
        raise OpenPITimeoutError(
            f"OpenPI request {request.request_id!r} deadline has already expired"
        )
    return min(endpoint_deadline, request_deadline_s)


def inference_payload(
    request: InferenceRequest,
    default_session_id: str,
) -> tuple[dict[str, Any], str]:
    request_payload = request.payload
    if isinstance(request_payload, RawRequest):
        payload = dict(request_payload.observation)
        if request_payload.prompt is not None:
            payload.setdefault("prompt", request_payload.prompt)
    elif isinstance(request_payload, Mapping):
        payload = dict(request_payload)
    else:
        raise TypeError("OpenPI request payload must be RawRequest or a mapping observation")

    metadata_session_id = request.metadata.get("session_id")
    if metadata_session_id is not None and not isinstance(metadata_session_id, str):
        raise TypeError("request metadata session_id must be a string")
    payload_session_id = payload.get("session_id")
    if payload_session_id is not None and not isinstance(payload_session_id, str):
        raise TypeError("OpenPI payload session_id must be a string")
    selected_session_id = payload_session_id or metadata_session_id or default_session_id
    if not selected_session_id:
        raise ValueError("OpenPI session_id must not be empty")

    payload["endpoint"] = "infer"
    payload["session_id"] = selected_session_id
    return payload, selected_session_id


def decode_inference_response(
    response: Any,
    *,
    expected_batch_size: int | None,
    expected_action_horizon: int | None,
    expected_action_dim: int | None,
) -> tuple[Any, dict[str, Any]]:
    if isinstance(response, Mapping) and (response.get("type") == "error" or "error" in response):
        message = response.get("message", response.get("error", response))
        raise OpenPIRemoteError(f"OpenPI inference server error: {message}")

    response_metadata: dict[str, Any] = {}
    if isinstance(response, Mapping) and "actions" in response:
        actions = response["actions"]
        for key, value in response.items():
            if key != "actions":
                response_metadata[str(key)] = value
    else:
        actions = response

    validated = validate_openpi_actions(
        actions,
        expected_batch_size=expected_batch_size,
        expected_action_horizon=expected_action_horizon,
        expected_action_dim=expected_action_dim,
    )
    return validated, response_metadata


def decode_reset_response(response: Any) -> str:
    if isinstance(response, Mapping) and (response.get("type") == "error" or "error" in response):
        message = response.get("message", response.get("error", response))
        raise OpenPIRemoteError(f"OpenPI reset server error: {message}")
    if isinstance(response, str):
        status = response
    elif isinstance(response, Mapping) and isinstance(response.get("status"), str):
        status = response["status"]
    else:
        raise OpenPIProtocolError(f"unexpected OpenPI reset response: {response!r}")
    if status.strip().lower() not in {"reset successful", "reset", "ok"}:
        raise OpenPIProtocolError(f"OpenPI reset was not acknowledged: {status!r}")
    return status
