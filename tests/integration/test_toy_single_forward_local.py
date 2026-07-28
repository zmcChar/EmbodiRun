from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.apps.local_toy_single_forward import (  # noqa: E402
    run_toy_single_forward,
)


def test_single_forward_adapter_runs_through_engine_and_backend() -> None:
    summary = run_toy_single_forward(
        target=(1.0, -2.0, 0.5),
        action_horizon=2,
        device="cpu",
        dtype="float32",
        mode="eager",
    )

    assert summary["model_id"] == "toy-single-forward"
    assert summary["plan_kind"] == "single_forward"
    assert summary["backend"] == "torch_cuda"
    assert summary["runtime_mode"] == "eager"
    assert summary["compile_failures"] == {}
    assert summary["shape"] == (2, 3)
    torch.testing.assert_close(
        summary["actions"],
        torch.tensor([[1.0, -2.0, 0.5], [1.0, -2.0, 0.5]]),
    )
