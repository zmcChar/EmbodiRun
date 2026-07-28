from __future__ import annotations

import pytest

from embodied_runtime.contracts import ModelAdapter, RawRequest, SingleForwardPlan
from embodied_runtime.models import get_model_adapter

torch = pytest.importorskip("torch")


def test_single_forward_package_is_not_a_flow_recipe() -> None:
    adapter = get_model_adapter(
        "toy_single_forward",
        action_horizon=2,
        action_dim=3,
    )
    assert isinstance(adapter, ModelAdapter)
    package = adapter.build_package()
    assert isinstance(package.plan, SingleForwardPlan)
    assert package.plan.required_entrypoints() == ("forward",)
    with pytest.raises(AttributeError, match="iterative-flow"):
        _ = package.recipe


def test_single_forward_adapter_cardinality_and_action_semantics() -> None:
    adapter = get_model_adapter(
        "toy_single_forward",
        action_horizon=2,
        action_dim=3,
    )
    package = adapter.build_package()
    one = adapter.preprocess_one(RawRequest(observation={"state": [1.0, 2.0, 3.0]}))
    two = adapter.preprocess_one(RawRequest(observation={"target": [-1.0, 0.0, 1.0]}))
    assert one["target"].shape == (1, 3)
    assert two["target"].shape == (1, 3)

    batch = adapter.collate((one, two))
    output = package.entrypoints[package.plan.forward](batch)
    samples = adapter.unbatch(output, batch_size=2)
    chunks = tuple(adapter.postprocess_one(sample) for sample in samples)

    assert len(chunks) == 2
    assert chunks[0].actions.shape == (2, 3)
    torch.testing.assert_close(
        chunks[0].actions,
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    )
    torch.testing.assert_close(
        chunks[1].actions,
        torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]),
    )
