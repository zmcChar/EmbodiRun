"""Wire request decoding and result encoding for the multi-robot cloud."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Any

from embodied_runtime.distributed.communication import (
    MULTI_ROBOT_PROTOCOL_VERSION,
    json_compatible,
)
from embodied_runtime.distributed.multitenant import MultiTenantInferenceService
from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.models.request import RawRequest


async def dispatch_multi_robot_request(
    service: MultiTenantInferenceService,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode one protocol request, dispatch it, and encode the response."""

    if request.get("protocol_version") != MULTI_ROBOT_PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {MULTI_ROBOT_PROTOCOL_VERSION}")
    kind = request.get("kind")
    if kind == "ping":
        return {
            "ok": True,
            "kind": "pong",
            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
            **provider_description(service),
            "registered_sessions": service.registered_session_count,
        }

    identity = RobotSessionIdentity.from_wire(request)
    if kind == "register":
        metadata = request.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("registration metadata must be an object")
        snapshot = service.register(identity, metadata=metadata)
        return {
            "ok": True,
            "kind": "registered",
            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
            **identity.to_wire(),
            **provider_description(service),
            "last_sequence_id": snapshot.last_sequence_id,
            "registered_sessions": service.registered_session_count,
        }
    if kind == "unregister":
        service.unregister(identity)
        return {
            "ok": True,
            "kind": "unregistered",
            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
            **identity.to_wire(),
            "registered_sessions": service.registered_session_count,
        }
    if kind != "infer":
        raise ValueError(f"unsupported request kind: {kind!r}")

    request_id = str(request.get("request_id") or "")
    if not request_id:
        raise ValueError("infer request requires request_id")
    sequence_id = int(request.get("sequence_id", 0))
    observation_id = str(request.get("observation_id") or "")
    observation_timestamp_s = float(request.get("observation_timestamp_s", -1.0))
    observation = request.get("observation")
    if not isinstance(observation, Mapping):
        raise TypeError("infer request observation must be an object")
    request_metadata = request.get("request_metadata", {})
    raw_metadata = request.get("raw_metadata", {})
    if not isinstance(request_metadata, Mapping):
        raise TypeError("request_metadata must be an object")
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("raw_metadata must be an object")
    prompt = request.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise TypeError("prompt must be a string or null")

    deadline = request.get("deadline_s")
    deadline_s = None if deadline is None else float(deadline)
    if deadline_s is not None and (not math.isfinite(deadline_s) or deadline_s <= 0):
        raise ValueError("deadline_s must be finite and greater than zero")
    if request.get("num_steps") is not None:
        raise ValueError(
            "remote num_steps overrides are not admitted; configure the cloud Provider instead"
        )
    seed_value = request.get("seed")
    seed = None if seed_value is None else int(seed_value)

    result = await service.infer_async(
        identity,
        InferenceRequest(
            payload=RawRequest(
                observation=dict(observation),
                prompt=prompt,
                metadata=dict(raw_metadata),
            ),
            request_id=request_id,
            deadline_s=deadline_s,
            num_steps=None,
            seed=seed,
            metadata=dict(request_metadata),
        ),
        sequence_id=sequence_id,
        observation_id=observation_id,
        observation_timestamp_s=observation_timestamp_s,
    )
    output, metadata = await asyncio.gather(
        asyncio.to_thread(json_compatible, result.output, path="result.output"),
        asyncio.to_thread(json_compatible, result.metadata, path="result.metadata"),
    )
    return {
        "ok": True,
        "kind": "inference_result",
        "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
        **identity.to_wire(),
        "request_id": result.request_id,
        "sequence_id": sequence_id,
        "observation_id": observation_id,
        "status": result.status.value,
        "output": output,
        "queue_time_s": result.queue_time_s,
        "execution_time_s": result.execution_time_s,
        "metadata": metadata,
    }


def provider_description(service: MultiTenantInferenceService) -> dict[str, Any]:
    capabilities = service.capabilities
    device = capabilities.device
    return {
        "provider": capabilities.name,
        "provider_runtime": capabilities.runtime,
        "model_id": capabilities.model.model_id,
        "model_family": capabilities.model.family,
        "model_revision": capabilities.model.revision,
        "action_space_id": service.contract.action_space_id,
        "supported_embodiments": sorted(service.contract.supported_embodiments),
        "action_dim": service.contract.action_dim,
        "action_horizon": service.contract.action_horizon,
        "backend": device.backend if device is not None else None,
        "device": device.device_id if device is not None else None,
    }
