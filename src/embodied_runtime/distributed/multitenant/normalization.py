"""Boundary validation and envelope normalization for shared inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.engine.provider import ProviderCapabilities
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult
from embodied_runtime.models.request import RawRequest

from .errors import SessionRegistrationError
from .types import SharedModelContract


def normalize_registration(
    contract: SharedModelContract,
    identity: RobotSessionIdentity,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if identity.action_space_id != contract.action_space_id:
        raise SessionRegistrationError(
            f"action_space_id {identity.action_space_id!r} is incompatible "
            f"with cloud contract {contract.action_space_id!r}"
        )
    if identity.embodiment not in contract.supported_embodiments:
        supported = ", ".join(sorted(contract.supported_embodiments))
        raise SessionRegistrationError(
            f"embodiment {identity.embodiment!r} is unsupported; cloud supports: {supported}"
        )
    normalized = dict(metadata or {})
    expected = {
        "edge_action_dim": contract.action_dim,
        "edge_action_horizon": contract.action_horizon,
    }
    for name, value in expected.items():
        if value is None:
            continue
        supplied = normalized.get(name)
        if supplied is None:
            raise SessionRegistrationError(f"registration metadata requires {name}")
        if not isinstance(supplied, int) or isinstance(supplied, bool) or supplied != value:
            raise SessionRegistrationError(
                f"{name} {supplied!r} is incompatible with cloud value {value}"
            )
    return normalized


def validate_inference_arguments(
    request: InferenceRequest,
    *,
    sequence_id: int,
    observation_id: str,
    observation_timestamp_s: float,
) -> str:
    if not isinstance(request, InferenceRequest):
        raise TypeError("request must be an InferenceRequest")
    if sequence_id <= 0:
        raise ValueError("sequence_id must be greater than zero")
    normalized_observation_id = observation_id.strip()
    if not normalized_observation_id:
        raise ValueError("observation_id must not be empty")
    if not math.isfinite(observation_timestamp_s) or observation_timestamp_s < 0:
        raise ValueError("observation_timestamp_s must be finite and non-negative")
    return normalized_observation_id


def prepare_provider_request(
    identity: RobotSessionIdentity,
    request: InferenceRequest,
    *,
    sequence_id: int,
    observation_id: str,
    observation_timestamp_s: float,
) -> tuple[str, InferenceRequest]:
    external_request_id = request.request_id
    internal_request_id = identity.namespace_request_id(external_request_id)
    authoritative_metadata = {
        **request.metadata,
        **identity.to_wire(),
        "sequence_id": sequence_id,
        "observation_id": observation_id,
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
                "observation_id": observation_id,
                "observation_timestamp_s": observation_timestamp_s,
            },
        )
    provider_request = replace(
        request,
        payload=payload,
        request_id=internal_request_id,
        # Remote priority is intentionally not trusted by the shared service.
        priority=0,
        metadata=authoritative_metadata,
    )
    return external_request_id, provider_request


def restore_external_result(
    capabilities: ProviderCapabilities,
    identity: RobotSessionIdentity,
    result: InferenceResult,
    *,
    external_request_id: str,
    sequence_id: int,
    observation_id: str,
    observation_timestamp_s: float,
) -> InferenceResult:
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
            "observation_id": observation_id,
            "observation_timestamp_s": observation_timestamp_s,
            "provider_request_id": result.request_id,
        },
    )


__all__ = [
    "normalize_registration",
    "prepare_provider_request",
    "restore_external_result",
    "validate_inference_arguments",
]
