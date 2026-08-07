"""Robot-edge request snapshots and cloud response validation."""

from __future__ import annotations

import copy
import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult
from embodied_runtime.models.request import RawRequest


@dataclass(frozen=True, slots=True)
class PreparedEdgeRequest:
    request: InferenceRequest
    raw: RawRequest
    metadata: dict[str, Any]

    def cloud_request(
        self,
        observation: Mapping[str, Any],
        *,
        deadline_s: float,
    ) -> InferenceRequest:
        return replace(
            self.request,
            payload=replace(
                self.raw,
                observation=observation,
                metadata=dict(self.metadata),
            ),
            deadline_s=deadline_s,
        )


def prepare_edge_request(
    identity: RobotSessionIdentity,
    observation: Mapping[str, Any],
    *,
    sequence_id: int,
    observation_timestamp_s: float,
    prompt: str | None,
) -> PreparedEdgeRequest:
    request_id = uuid.uuid4().hex
    metadata = {
        **identity.to_wire(),
        "sequence_id": sequence_id,
        "observation_id": f"{identity.session_id}-obs-{sequence_id}",
        "observation_timestamp_s": observation_timestamp_s,
    }
    raw = RawRequest(
        observation=dict(observation),
        prompt=prompt,
        metadata=metadata,
    )
    return PreparedEdgeRequest(
        request=InferenceRequest(
            payload=raw,
            request_id=request_id,
            seed=session_sequence_seed(identity.session_id, sequence_id),
            metadata=metadata,
        ),
        raw=raw,
        metadata=metadata,
    )


def validate_cloud_registration(
    identity: RobotSessionIdentity,
    edge_capabilities: Any,
    response: Mapping[str, Any],
    *,
    last_sequence_id: int,
) -> None:
    expected = {
        "action_space_id": identity.action_space_id,
        "action_dim": edge_capabilities.model.action_dim,
        "action_horizon": edge_capabilities.model.action_horizon,
    }
    for name, value in expected.items():
        if response.get(name) != value:
            raise ValueError(
                f"cloud {name} {response.get(name)!r} does not match edge value {value!r}"
            )
    supported = response.get("supported_embodiments")
    if not isinstance(supported, list) or identity.embodiment not in supported:
        raise ValueError(f"cloud does not support embodiment {identity.embodiment!r}")
    server_sequence = int(response.get("last_sequence_id", 0))
    if server_sequence > last_sequence_id:
        raise ValueError(
            f"cloud session is already at sequence {server_sequence}; "
            "rotate session_id instead of reusing an older session"
        )


def validate_cloud_result(
    identity: RobotSessionIdentity,
    result: InferenceResult,
    *,
    sequence_id: int,
) -> None:
    for name, expected in identity.to_wire().items():
        if result.metadata.get(name) != expected:
            raise RuntimeError(f"cloud result has mismatched {name}")
    source_sequence = int(result.metadata.get("sequence_id", 0))
    if not 0 < source_sequence <= sequence_id:
        raise RuntimeError(
            f"cloud result sequence {source_sequence} is invalid for edge sequence {sequence_id}"
        )


def snapshot_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Take ownership of mutable frame storage without serializing it."""

    return {key: _snapshot_value(value) for key, value in observation.items()}


def _snapshot_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    clone = getattr(candidate, "clone", None)
    if callable(clone):
        return clone()
    copy_value = getattr(candidate, "copy", None)
    if callable(copy_value):
        return copy_value()
    return copy.deepcopy(candidate)


def session_sequence_seed(session_id: str, sequence_id: int) -> int:
    """Return a stable seed without relying on Python's randomized hash."""

    digest = hashlib.blake2b(
        f"{session_id}:{sequence_id}".encode(),
        digest_size=8,
        person=b"edge-cloud-seed",
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)
