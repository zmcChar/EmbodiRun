from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.backends.compile import CompileOptions
from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models import available_models, get_model_adapter
from embodied_runtime.models.interfaces import ModelAdapter
from embodied_runtime.models.plans.single_forward import SingleForwardPlan
from embodied_runtime.models.request import RawRequest
from embodied_runtime.models.vla.openvla_oft import OpenVLAOFTAdapter


class _InjectedProcessor:
    """Small model-owned frontend whose leaves retain the required B=1 axis."""

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    def preprocess_one(self, request: RawRequest):
        self.prompts.append(request.prompt)
        token = int(request.observation.get("token", 0))
        return {
            "input_ids": torch.tensor([[1, token]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.full(
                (1, 3, 2, 2),
                float(token),
                dtype=torch.float32,
            ),
        }


class _InjectedModule(torch.nn.Module):
    """A deterministic OFT-shaped module suitable for backend/engine tests."""

    def __init__(self, action_horizon: int = 3, action_dim: int = 2) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.n_tokens = action_horizon * action_dim
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        offsets = input_ids[:, -1].to(self.scale.dtype).reshape(batch_size, 1, 1)
        template = torch.arange(
            self.n_tokens,
            device=self.scale.device,
            dtype=self.scale.dtype,
        ).reshape(1, self.action_horizon, self.action_dim)
        actions = (template + offsets) * self.scale
        action_tokens = torch.arange(
            self.n_tokens,
            device=self.scale.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        return {
            "actions": actions,
            "action_tokens": action_tokens,
        }


def _injected_package():
    adapter = OpenVLAOFTAdapter()
    processor = _InjectedProcessor()
    module = _InjectedModule()
    package = adapter.build_package(
        "injected://openvla-oft",
        module=module,
        processor=processor,
        action_dim=module.action_dim,
        action_horizon=module.action_horizon,
    )
    return adapter, processor, module, package


def _cpu_device(backend: TorchCudaBackend):
    return next(device for device in backend.probe() if device.device_id == "cpu")


def test_openvla_oft_registry_and_description_are_lazy() -> None:
    assert "openvla_oft" in available_models()

    adapter = get_model_adapter("openvla_oft")

    assert isinstance(adapter, ModelAdapter)
    assert isinstance(adapter, OpenVLAOFTAdapter)
    spec = adapter.describe()
    assert spec.model_id == "openvla-oft"
    assert spec.family == "openvla_oft_categorical"
    assert spec.modalities == ("vision", "language")
    assert spec.action_dim == 7
    assert spec.action_horizon == 8


def test_injected_package_uses_single_forward_and_preserves_batch_semantics() -> None:
    adapter, processor, module, package = _injected_package()

    assert isinstance(package.plan, SingleForwardPlan)
    assert package.plan.required_entrypoints() == ("forward",)
    assert set(package.entrypoints) == {"forward"}
    assert package.metadata["runtime_module"] is module
    assert package.spec.action_dim == 2
    assert package.spec.action_horizon == 3

    first = adapter.preprocess_one(
        RawRequest(observation={"token": 4}, prompt="move the first object")
    )
    second = adapter.preprocess_one(
        RawRequest(observation={"token": 9}, prompt="move the second object")
    )
    assert first["input_ids"].shape == (1, 2)
    assert first["attention_mask"].shape == (1, 2)
    assert first["pixel_values"].shape == (1, 3, 2, 2)
    assert processor.prompts == ["move the first object", "move the second object"]

    payload = adapter.collate((first, second))
    assert payload["input_ids"].shape == (2, 2)
    assert payload["pixel_values"].shape == (2, 3, 2, 2)

    output = package.entrypoints[package.plan.forward](payload)
    assert output["actions"].shape == (2, 3, 2)
    assert output["action_tokens"].shape == (2, 6)

    samples = adapter.unbatch(output, batch_size=2)
    chunks = tuple(adapter.postprocess_one(sample) for sample in samples)
    assert len(chunks) == 2
    assert chunks[0].actions.shape == (3, 2)
    assert chunks[1].actions.shape == (3, 2)
    torch.testing.assert_close(
        chunks[0].actions,
        torch.tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
    )
    torch.testing.assert_close(
        chunks[1].actions,
        torch.tensor([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]),
    )


def test_injected_package_runs_through_cpu_backend_and_engine() -> None:
    adapter, _, _, package = _injected_package()
    payload = adapter.preprocess_one(
        RawRequest(observation={"token": 3}, prompt="pick up the block")
    )

    backend = TorchCudaBackend()
    session = backend.load(
        backend.compile(
            package,
            _cpu_device(backend),
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
        result = engine.infer(payload)
        chunk = adapter.postprocess_one(result.output)
    finally:
        engine.close()

    assert result.metadata["model_id"] == "openvla-oft"
    assert result.metadata["backend"] == "torch_cuda"
    assert result.output["action_tokens"].shape == (6,)
    assert chunk.actions.shape == (3, 2)
    torch.testing.assert_close(
        chunk.actions,
        torch.tensor([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )


@pytest.mark.skipif(
    not os.environ.get("EMBODIED_RUNTIME_OPENVLA_OFT_CHECKPOINT"),
    reason="set EMBODIED_RUNTIME_OPENVLA_OFT_CHECKPOINT to a local OpenVLA-OFT checkpoint",
)
def test_real_local_checkpoint_builds_a_single_forward_package() -> None:
    """Opt-in loader check; intentionally does not assume a CUDA device is present."""

    checkpoint = os.environ["EMBODIED_RUNTIME_OPENVLA_OFT_CHECKPOINT"]
    adapter = get_model_adapter("openvla_oft")
    package = adapter.build_package(
        checkpoint,
        local_files_only=True,
        load_device="cpu",
        load_dtype="bfloat16",
    )

    assert isinstance(package.plan, SingleForwardPlan)
    assert package.metadata["runtime_module"] is not None
    assert package.spec.action_dim == 7
    assert package.spec.action_horizon == 8

    payload = adapter.preprocess_one(
        RawRequest(
            observation={"image": torch.zeros(3, 224, 224)},
            prompt="pick up the block",
        )
    )
    assert set(payload) == {"input_ids", "attention_mask", "pixel_values"}
    assert all(value.shape[0] == 1 for value in payload.values())
