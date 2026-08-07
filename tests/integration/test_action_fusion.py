from __future__ import annotations

import pytest

from embodied_runtime.apps.action_fusion import (
    blend_action_results,
    make_action_result_fuser,
)
from embodied_runtime.engine import InferenceResult

torch = pytest.importorskip("torch")


def _result(
    request_id: str,
    actions,
    *,
    model_id: str,
    device_id: str,
) -> InferenceResult:
    return InferenceResult(
        request_id=request_id,
        output={"actions": actions},
        execution_time_s=0.1,
        metadata={"model_id": model_id, "device_id": device_id},
    )


def test_action_fusion_combines_both_results_on_the_edge_device() -> None:
    edge_actions = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
    cloud_actions = torch.tensor([[4.0, 6.0], [8.0, 10.0]])
    edge = _result("edge-2", edge_actions, model_id="small", device_id="cpu")
    cloud = _result("cloud-1", cloud_actions, model_id="large", device_id="cuda:0")

    fused = make_action_result_fuser(cloud_weight=0.25)(edge, cloud)

    torch.testing.assert_close(
        fused.output["actions"],
        torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
    )
    assert fused.output["actions"].device == edge_actions.device
    assert fused.metadata["coordination_mode"] == "async_blend"
    assert fused.metadata["cloud_weight"] == 0.25
    assert fused.metadata["edge_model_id"] == "small"
    assert fused.metadata["cloud_model_id"] == "large"
    assert fused.request_id == "edge-2"
    torch.testing.assert_close(edge_actions, torch.tensor([[0.0, 2.0], [4.0, 6.0]]))
    torch.testing.assert_close(cloud_actions, torch.tensor([[4.0, 6.0], [8.0, 10.0]]))


def test_action_fusion_rejects_incompatible_shapes() -> None:
    edge = _result(
        "edge",
        torch.zeros(2, 3),
        model_id="small",
        device_id="cpu",
    )
    cloud = _result(
        "cloud",
        torch.zeros(4, 3),
        model_id="large",
        device_id="cuda:0",
    )

    with pytest.raises(ValueError, match="action shapes must match"):
        blend_action_results(edge, cloud, cloud_weight=0.5)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_action_fusion_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        make_action_result_fuser(cloud_weight=weight)
