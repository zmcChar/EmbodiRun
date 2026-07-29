"""Action-level composition helpers for cloud/edge integration apps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from embodied_runtime.contracts import InferenceResult


def make_action_result_fuser(
    *,
    cloud_weight: float,
) -> Callable[[InferenceResult, InferenceResult], InferenceResult]:
    """Build a synchronous result fuser for ``AsyncFailoverCoordinator``."""

    _validate_weight(cloud_weight)

    def fuse(edge: InferenceResult, cloud: InferenceResult) -> InferenceResult:
        return blend_action_results(
            edge,
            cloud,
            cloud_weight=cloud_weight,
        )

    return fuse


def blend_action_results(
    edge: InferenceResult,
    cloud: InferenceResult,
    *,
    cloud_weight: float,
) -> InferenceResult:
    """Blend a current edge action with the latest cloud action on the edge device.

    Only the compact action result crosses the device/link boundary. The cloud
    tensor is detached and copied to the edge tensor's device and dtype before
    fusion; model parameters and hidden states remain at their original
    placements.
    """

    _validate_weight(cloud_weight)
    edge_actions = _actions_from(edge.output, source="edge")
    cloud_actions = _actions_from(cloud.output, source="cloud")
    if tuple(edge_actions.shape) != tuple(cloud_actions.shape):
        raise ValueError(
            "edge/cloud action shapes must match for blending: "
            f"edge={tuple(edge_actions.shape)}, cloud={tuple(cloud_actions.shape)}"
        )
    if not hasattr(cloud_actions, "detach") or not hasattr(cloud_actions, "to"):
        raise TypeError("cloud actions must support detach() and to()")
    if not hasattr(edge_actions, "device") or not hasattr(edge_actions, "dtype"):
        raise TypeError("edge actions must expose device and dtype")

    cloud_on_edge = cloud_actions.detach().to(
        device=edge_actions.device,
        dtype=edge_actions.dtype,
    )
    fused_actions = edge_actions + (cloud_on_edge - edge_actions) * cloud_weight
    output = (
        {**edge.output, "actions": fused_actions}
        if isinstance(edge.output, Mapping)
        else fused_actions
    )
    metadata = {
        **edge.metadata,
        "coordination_mode": "async_blend",
        "cloud_weight": cloud_weight,
        "edge_request_id": edge.request_id,
        "cloud_request_id": cloud.request_id,
        "edge_model_id": edge.metadata.get("model_id"),
        "cloud_model_id": cloud.metadata.get("model_id"),
        "edge_device_id": edge.metadata.get("device_id"),
        "cloud_device_id": cloud.metadata.get("device_id"),
        "cloud_execution_time_s": cloud.execution_time_s,
    }
    return InferenceResult(
        request_id=edge.request_id,
        output=output,
        status=edge.status,
        queue_time_s=edge.queue_time_s,
        execution_time_s=edge.execution_time_s,
        metadata=metadata,
    )


def _actions_from(output: Any, *, source: str) -> Any:
    if isinstance(output, Mapping):
        try:
            actions = output["actions"]
        except KeyError as error:
            raise ValueError(f"{source} output has no 'actions' entry") from error
    else:
        actions = output
    if not hasattr(actions, "shape"):
        raise TypeError(f"{source} actions must be tensor-like")
    return actions


def _validate_weight(cloud_weight: float) -> None:
    if not 0.0 <= cloud_weight <= 1.0:
        raise ValueError("cloud_weight must be between zero and one")


__all__ = ["blend_action_results", "make_action_result_fuser"]
