from __future__ import annotations

import pytest

from embodied_runtime.models import get_model_adapter
from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.request import RawRequest

torch = pytest.importorskip("torch")


def _run_flow(package, payload, *, noise, steps: int | None = None):
    prefix = package.entrypoints[package.plan.encode](payload)
    state = package.entrypoints[package.plan.initialize](
        {
            "batch_size": payload["target"].shape[0],
            "noise": noise,
            "seed": 999,
        }
    )
    for time_value, dt in package.plan.schedule(steps):
        velocity = package.entrypoints[package.plan.step](
            {
                "state": state,
                "time": time_value,
                "prefix": prefix,
            }
        )
        state = state + dt * velocity
    return package.entrypoints[package.plan.finalize]({"state": state})


def test_toy_package_exposes_four_staged_entrypoints() -> None:
    adapter = get_model_adapter("toy_flow", action_horizon=3, action_dim=2)
    package = adapter.build_package("")
    assert package.plan.required_entrypoints() == (
        "encode_prefix",
        "init_state",
        "denoise_step",
        "finalize",
    )
    assert package.metadata["step_output"] == "velocity"
    assert package.metadata["runtime_module"] is not None
    assert package.entrypoint_specs["denoise_step"].metadata["output"] == "velocity"


def test_toy_flow_runs_end_to_end_on_cpu() -> None:
    adapter = get_model_adapter("toy_flow", action_horizon=3, action_dim=2)
    package = adapter.build_package("")
    samples = (
        adapter.preprocess_one(RawRequest(observation={"target": [1.0, 2.0]})),
        adapter.preprocess_one(RawRequest(observation={"state": [-1.0, 0.5]})),
    )
    assert all(sample["target"].shape == (1, 2) for sample in samples)
    payload = adapter.collate(samples)
    assert payload["target"].shape == (2, 2)
    initial = torch.zeros(2, 3, 2)
    output = _run_flow(package, payload, noise=initial)
    samples = adapter.unbatch(output, batch_size=2)
    chunks = tuple(adapter.postprocess_one(sample) for sample in samples)
    assert len(chunks) == 2
    assert chunks[0].actions.shape == (3, 2)
    targets = payload["target"][:, None, :].expand_as(output["actions"])
    initial_error = torch.linalg.vector_norm(initial - targets)
    final_error = torch.linalg.vector_norm(output["actions"] - targets)
    assert final_error < initial_error


def test_collate_preserves_tuple_slots_and_unbatch_has_exact_cardinality() -> None:
    adapter = get_model_adapter("toy_flow")
    samples = (
        {"images": (torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2))},
        {"images": (torch.ones(1, 3, 2, 2), torch.zeros(1, 3, 2, 2))},
    )
    batch = adapter.collate(samples)
    assert isinstance(batch["images"], tuple)
    assert len(batch["images"]) == 2
    assert all(camera.shape == (2, 3, 2, 2) for camera in batch["images"])

    unbatched = adapter.unbatch({"actions": torch.zeros(2, 3, 2)}, batch_size=2)
    assert len(unbatched) == 2
    assert all(sample["actions"].shape == (3, 2) for sample in unbatched)

    with pytest.raises(ModelPackageError, match="leading dimension 3"):
        adapter.unbatch({"actions": torch.zeros(2, 3, 2)}, batch_size=3)


def test_init_state_prefers_explicit_noise_over_generator_and_seed() -> None:
    adapter = get_model_adapter("toy_flow")
    package = adapter.build_package("")
    noise = torch.full((1, 3, 2), 7.0)
    state = package.entrypoints["init_state"](
        {
            "batch_size": 1,
            "noise": noise,
            "generator": torch.Generator().manual_seed(1),
            "seed": 2,
        }
    )
    assert torch.equal(state, noise)


def test_seeded_initial_state_is_repeatable() -> None:
    adapter = get_model_adapter("toy_flow")
    package = adapter.build_package("")
    first = package.entrypoints["init_state"]({"batch_size": 2, "seed": 123})
    second = package.entrypoints["init_state"]({"batch_size": 2, "seed": 123})
    assert torch.equal(first, second)
