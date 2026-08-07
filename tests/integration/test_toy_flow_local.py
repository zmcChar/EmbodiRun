from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.apps.local_toy_flow import run_toy_flow


def test_model_package_runs_through_engine_and_torch_backend() -> None:
    summary = run_toy_flow(
        target=(0.25, -0.5),
        device="cpu",
        dtype="float32",
        mode="eager",
        num_steps=4,
        seed=7,
    )

    assert summary["model_id"] == "toy-flow"
    assert summary["backend"] == "torch_cuda"
    assert summary["device"] == "cpu"
    assert summary["runtime_mode"] == "eager"
    assert summary["compile_failures"] == {}
    assert summary["shape"] == (3, 2)
    assert torch.isfinite(summary["actions"]).all()


def test_toy_flow_is_repeatable_for_a_seed() -> None:
    options = {
        "target": (0.25, -0.5),
        "device": "cpu",
        "dtype": "float32",
        "mode": "eager",
        "num_steps": 4,
        "seed": 11,
    }
    first = run_toy_flow(**options)
    second = run_toy_flow(**options)

    torch.testing.assert_close(first["actions"], second["actions"])
