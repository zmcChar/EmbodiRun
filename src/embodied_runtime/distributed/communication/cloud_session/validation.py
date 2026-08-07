"""Validation of cloud-session bindings and response payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from embodied_runtime.distributed.session import RobotSessionIdentity

from .errors import CloudSessionProtocolError, CloudSessionRemoteError
from .protocol import MULTI_ROBOT_PROTOCOL_VERSION


def validate_bound_metadata(
    metadata: Mapping[str, Any],
    identity: RobotSessionIdentity,
) -> None:
    for name, expected in identity.to_wire().items():
        supplied = metadata.get(name)
        if supplied is not None and str(supplied) != expected:
            raise ValueError(
                f"request metadata cannot override bound {name}: {supplied!r} != {expected!r}"
            )


def nonnegative_timing(response: Mapping[str, Any], name: str) -> float:
    raw = response.get(name)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise CloudSessionProtocolError(f"cloud result {name} must be a number")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise CloudSessionProtocolError(f"cloud result {name} must be finite and non-negative")
    return value


def validate_registration_response(
    response: Mapping[str, Any],
    *,
    identity: RobotSessionIdentity,
    registration_metadata: Mapping[str, Any],
) -> None:
    supported = response.get("supported_embodiments")
    if (
        not isinstance(supported, list)
        or any(not isinstance(item, str) for item in supported)
        or identity.embodiment not in supported
    ):
        raise CloudSessionProtocolError("cloud registration has invalid supported_embodiments")

    for name, edge_name in (
        ("action_dim", "edge_action_dim"),
        ("action_horizon", "edge_action_horizon"),
    ):
        value = response.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CloudSessionProtocolError(f"cloud registration {name} must be a positive integer")
        edge_value = registration_metadata.get(edge_name)
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


def validate_action_output(output: Any, server_metadata: Mapping[str, Any]) -> None:
    """Reject malformed cloud actions before they can become authoritative."""

    if not isinstance(output, Mapping):
        raise CloudSessionProtocolError("cloud output must be an object")
    actions = output.get("actions")
    if not isinstance(actions, list) or not actions:
        raise CloudSessionProtocolError("cloud output.actions must be a non-empty array")

    action_dim = server_metadata.get("action_dim")
    action_horizon = server_metadata.get("action_horizon")
    if action_dim is None or action_horizon is None:
        raise CloudSessionProtocolError("cloud registration did not declare an action shape")
    expected_dim = int(action_dim)
    expected_horizon = int(action_horizon)

    if all(_is_finite_number(value) for value in actions):
        rows = [actions]
    elif all(isinstance(row, list) for row in actions):
        rows = actions
    else:
        raise CloudSessionProtocolError("cloud output.actions must be a numeric vector or matrix")

    if len(rows) != expected_horizon:
        raise CloudSessionProtocolError(
            "cloud output action horizon does not match the registered contract"
        )
    for row in rows:
        if len(row) != expected_dim or not all(_is_finite_number(value) for value in row):
            raise CloudSessionProtocolError(
                "cloud output action dimension or values do not match the registered contract"
            )


def validate_base_response(
    response: Mapping[str, Any],
    *,
    expected_kind: str,
    identity: RobotSessionIdentity,
) -> None:
    if response.get("ok") is not True:
        raise _response_error(response)
    if response.get("protocol_version") != MULTI_ROBOT_PROTOCOL_VERSION:
        raise CloudSessionProtocolError("cloud protocol version mismatch")
    if response.get("kind") != expected_kind:
        raise CloudSessionProtocolError(
            f"expected cloud response kind {expected_kind!r}, got {response.get('kind')!r}"
        )
    for name, expected in identity.to_wire().items():
        if response.get(name) != expected:
            raise CloudSessionProtocolError(
                f"cloud response {name} does not match the bound session"
            )


def validate_inference_response(
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


def _response_error(response: Mapping[str, Any]) -> CloudSessionRemoteError:
    error_type = str(response.get("error_type") or "CloudError")
    message = str(response.get("error") or "cloud request failed")
    return CloudSessionRemoteError(error_type, message)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
