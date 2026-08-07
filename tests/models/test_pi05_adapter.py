from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.backends.compile import CompileOptions
from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models import get_model_adapter
from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.interfaces import ModelAdapter
from embodied_runtime.models.vla.pi05 import Pi05Adapter


class _InjectedPolicy(torch.nn.Module):
    """Enough surface to verify package construction without allocating 4.14B weights."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            max_action_dim=32,
            chunk_size=50,
            num_inference_steps=10,
            output_features={"action": SimpleNamespace(shape=(7,))},
            rtc_config=None,
            image_features={
                "observation.images.base": object(),
                "observation.images.left_wrist": object(),
                "observation.images.right_wrist": object(),
            },
            image_resolution=(224, 224),
        )

    def _preprocess_images(self, batch):
        raise AssertionError("not used by this package-construction test")


def test_pi05_adapter_is_lazy_and_describes_expected_family() -> None:
    adapter = get_model_adapter("pi05")
    assert isinstance(adapter, ModelAdapter)
    assert isinstance(adapter, Pi05Adapter)
    spec = adapter.describe()
    assert spec.family == "pi05_flow"
    assert spec.action_dim == 32
    assert spec.action_horizon == 50


def test_injected_policy_builds_package_without_backend_imports() -> None:
    pytest.importorskip("lerobot")
    adapter = get_model_adapter("pi05")
    package = adapter.build_package("", policy=_InjectedPolicy())
    assert package.plan.kind == "iterative_flow"
    assert package.plan.default_num_steps == 10
    assert package.metadata["step_output"] == "velocity"
    assert package.metadata["runtime_module"] is not None
    assert set(package.entrypoints) == set(package.plan.required_entrypoints())
    assert package.spec.action_horizon == 50
    assert package.spec.action_dim == 7

    outputs = {"actions": torch.zeros(1, 50, 32)}
    unbatched = adapter.unbatch(outputs, batch_size=1)
    assert len(unbatched) == 1
    assert adapter.postprocess_one(unbatched[0]).actions.shape == (50, 7)


def test_rtc_enabled_checkpoint_is_rejected() -> None:
    policy = _InjectedPolicy()
    policy.config.rtc_config = SimpleNamespace(enabled=True)
    adapter = get_model_adapter("pi05")
    with pytest.raises(ModelPackageError, match="RTC-enabled"):
        adapter.build_package("", policy=policy)


@pytest.mark.skipif(
    not os.environ.get("EMBODIED_RUNTIME_PI05_CHECKPOINT"),
    reason="set EMBODIED_RUNTIME_PI05_CHECKPOINT to a local LeRobot pi05 checkpoint",
)
def test_real_local_checkpoint_loads_and_matches_reference() -> None:
    """Opt-in real weights plus exact staged and Engine/Backend parity."""

    checkpoint = os.environ["EMBODIED_RUNTIME_PI05_CHECKPOINT"]
    adapter = get_model_adapter("pi05")
    package = adapter.build_package(
        checkpoint,
        local_files_only=True,
        load_device="cpu",
        load_dtype="bfloat16",
    )
    payload = adapter.synthetic_batch(batch_size=1, language_length=8, seed=0)
    assert payload["tokens"].shape == (1, 8)
    module = package.metadata["runtime_module"]
    assert module is not None

    if not torch.cuda.is_available():
        return

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to("cuda:0")
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    backend = TorchCudaBackend()
    cuda_device = next(device for device in backend.probe() if device.device_id == "cuda:0")
    session = backend.load(
        backend.compile(
            package,
            cuda_device,
            CompileOptions(mode="eager", dtype=None),
        )
    )
    engine = ExecutionEngine(
        package,
        session,
        batcher=adapter.collate,
        splitter=adapter.unbatch,
    )
    try:
        payload = move(payload)
        generator = torch.Generator(device="cuda:0").manual_seed(23)
        noise = torch.randn(
            1,
            package.spec.action_horizon,
            int(package.metadata["runtime_module"].config.max_action_dim),
            generator=generator,
            device="cuda:0",
            dtype=torch.float32,
        )
        with torch.inference_mode():
            expected = adapter.loaded_policy.model.sample_actions(
                payload["images"],
                payload["image_masks"],
                payload["tokens"],
                payload["token_masks"],
                noise=noise.clone(),
                num_steps=10,
            )
            prefix = package.entrypoints[package.plan.encode](payload)
            state = package.entrypoints[package.plan.initialize](
                {"batch_size": 1, "noise": noise.clone()}
            )
            for time_value, dt in package.plan.schedule(10):
                velocity = package.entrypoints[package.plan.step](
                    {"state": state, "time": time_value, "prefix": prefix}
                )
                state = state + dt * velocity
            staged = package.entrypoints[package.plan.finalize]({"state": state})["actions"]
            runtime = engine.infer(payload, num_steps=10, seed=23).output["actions"]

        torch.testing.assert_close(staged, expected, rtol=0, atol=0)
        torch.testing.assert_close(runtime, expected[0], rtol=0, atol=0)
    finally:
        engine.close()
