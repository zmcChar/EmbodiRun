from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import (
    CompileOptions,
    ModelAdapter,
    ModelPackageError,
    RawRequest,
)
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models import available_models, get_model_adapter
from embodied_runtime.models.vla.smolvla import SmolVLAAdapter
from embodied_runtime.models.vla.smolvla import adapter as smolvla_adapter_module
from embodied_runtime.models.vla.smolvla.modeling_smolvla import (
    LANGUAGE_ATTENTION_MASK,
    LANGUAGE_TOKENS,
    SmolVLAFlowPlan,
)
from embodied_runtime.models.vla.smolvla.processing_smolvla import OBSERVATION_STATE

torch = pytest.importorskip("torch")

_BASE_IMAGE = "observation.images.base"
_WRIST_IMAGE = "observation.images.wrist"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def __call__(
        self,
        texts: list[str],
        *,
        padding: str,
        padding_side: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding == "max_length"
        assert padding_side == "right"
        assert truncation
        assert return_tensors == "pt"
        self.seen_texts.extend(texts)
        batch_size = len(texts)
        return {
            "input_ids": torch.arange(max_length).expand(batch_size, -1).clone(),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
        }


class _FakeVLMWithExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefix_calls = 0

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del attention_mask, position_ids, past_key_values
        assert use_cache
        assert fill_kv_cache
        prefix = inputs_embeds[0]
        assert prefix is not None
        self.prefix_calls += 1
        return prefix, {"encoded_prefix": prefix.mean(dim=1)}


class _FakeSmolVLAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_in_proj = torch.nn.Linear(4, 4, bias=False)
        self.vlm_with_expert = _FakeVLMWithExpert()
        self.step_times: list[torch.Tensor] = []

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
        *,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(images) == len(image_masks) == 2
        assert language_tokens.shape == language_masks.shape
        prefix = torch.stack((state, state + 1.0), dim=1)
        batch_size = state.shape[0]
        pad_masks = torch.ones(batch_size, 2, dtype=torch.bool, device=state.device)
        block_masks = torch.tensor(
            (0, 1),
            dtype=torch.long,
            device=state.device,
        ).expand(batch_size, -1)
        return prefix, pad_masks, block_masks

    def denoise_step(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict[str, torch.Tensor],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        assert prefix_pad_masks.shape == (x_t.shape[0], 2)
        assert past_key_values["encoded_prefix"].shape == (x_t.shape[0], 4)
        self.step_times.append(timestep.detach().cpu().clone())
        return torch.full_like(x_t, 2.0)


class _FakeSmolVLAPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def feature(shape: tuple[int, ...]) -> SimpleNamespace:
            return SimpleNamespace(shape=shape)

        self.config = SimpleNamespace(
            image_features={
                _BASE_IMAGE: feature((3, 4, 4)),
                _WRIST_IMAGE: feature((3, 4, 4)),
            },
            robot_state_feature=feature((3,)),
            action_feature=feature((2,)),
            output_features={"action": feature((2,))},
            max_action_dim=4,
            chunk_size=3,
            num_steps=2,
            tokenizer_max_length=6,
            use_cache=True,
            rtc_config=None,
            adapt_to_pi_aloha=False,
        )
        self.language_tokenizer = _FakeTokenizer()
        self.model = _FakeSmolVLAModel()
        self.prepare_batch_calls = 0
        self.unnormalize_calls = 0

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.prepare_batch_calls += 1
        return batch

    def prepare_images(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        images = [batch[name] for name in self.config.image_features]
        batch_size = images[0].shape[0]
        masks = [torch.ones(batch_size, dtype=torch.bool, device=image.device) for image in images]
        return images, masks

    def prepare_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = batch[OBSERVATION_STATE]
        padding = self.config.max_action_dim - state.shape[-1]
        return torch.nn.functional.pad(state, (0, padding))

    def unnormalize_outputs(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.unnormalize_calls += 1
        return {"action": outputs["action"] * 3.0 + 5.0}


def _request(*, prompt: str = "pick up the block") -> RawRequest:
    return RawRequest(
        observation={
            "images": torch.stack(
                (
                    torch.full((3, 4, 4), 0.25),
                    torch.full((3, 4, 4), 0.75),
                )
            ),
            "state": torch.tensor((0.1, 0.2, 0.3)),
        },
        prompt=prompt,
    )


def _cpu_device(backend: TorchCudaBackend):
    return next(device for device in backend.probe() if device.device_id == "cpu")


def test_smolvla_registry_and_description_are_lazy() -> None:
    assert "smolvla" in available_models()

    adapter = get_model_adapter("smolvla")

    assert isinstance(adapter, ModelAdapter)
    assert isinstance(adapter, SmolVLAAdapter)
    spec = adapter.describe()
    assert spec.family == "smolvla_flow"
    assert spec.action_dim == 6
    assert spec.action_horizon == 50


def test_smolvla_flow_plan_matches_legacy_float32_time_accumulation() -> None:
    schedule = SmolVLAFlowPlan(default_num_steps=10).schedule()

    assert schedule[1] == (0.8999999761581421, -0.10000000149011612)
    assert schedule[2][0] == 0.7999999523162842
    assert schedule[-1][0] == 0.09999992698431015


def test_injected_policy_runs_staged_flow_through_cpu_backend_and_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_loader(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("injected-policy tests must not invoke the LeRobot loader")

    monkeypatch.setattr(smolvla_adapter_module, "load_lerobot_smolvla", reject_loader)
    policy = _FakeSmolVLAPolicy()
    adapter = SmolVLAAdapter()
    package = adapter.build_package("", policy=policy)
    payload = adapter.preprocess_one(_request())

    assert package.plan.kind == "iterative_flow"
    assert package.plan.default_num_steps == 2
    assert set(package.entrypoints) == set(package.plan.required_entrypoints())
    assert package.metadata["runtime_module"].lerobot_policy is policy
    assert package.metadata["action_contract"] == {
        "native_dim": 2,
        "internal_padded_dim": 4,
        "numeric_fusion_with_pi05": False,
    }
    assert package.spec.action_horizon == 3
    assert package.spec.action_dim == 2

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
    seed = 17
    try:
        result = engine.infer(payload, num_steps=2, seed=seed)
        chunk = adapter.postprocess_one(result.output)
    finally:
        engine.close()

    expected_noise = torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    # Two Euler steps use dt=-0.5. The fake model returns velocity=2, then
    # finalize crops 4->2 action dimensions and applies x * 3 + 5.
    expected = ((expected_noise[..., :2] - 2.0) * 3.0 + 5.0)[0]
    assert result.output["actions"].shape == (3, 2)
    assert chunk.actions.shape == (3, 2)
    torch.testing.assert_close(chunk.actions, expected)
    assert policy.prepare_batch_calls == 1
    assert policy.model.vlm_with_expert.prefix_calls == 1
    assert policy.unnormalize_calls == 1
    assert len(policy.model.step_times) == 2
    torch.testing.assert_close(policy.model.step_times[0], torch.tensor((1.0,)))
    torch.testing.assert_close(policy.model.step_times[1], torch.tensor((0.5,)))


def test_package_can_override_default_flow_steps_for_cpu_smoke() -> None:
    adapter = SmolVLAAdapter()

    package = adapter.build_package(
        "",
        policy=_FakeSmolVLAPolicy(),
        default_num_steps=1,
    )

    assert package.plan.default_num_steps == 1


def test_package_rejects_non_positive_default_flow_steps() -> None:
    adapter = SmolVLAAdapter()

    with pytest.raises(ValueError, match="default_num_steps must be greater than zero"):
        adapter.build_package(
            "",
            policy=_FakeSmolVLAPolicy(),
            default_num_steps=0,
        )


def test_processor_retains_batch_one_and_collate_owns_batching() -> None:
    adapter = SmolVLAAdapter()
    policy = _FakeSmolVLAPolicy()
    adapter.build_package("", policy=policy)

    first = adapter.preprocess_one(_request(prompt="first task"))
    second = adapter.preprocess_one(_request(prompt="second task"))

    assert set(first) == {
        _BASE_IMAGE,
        _WRIST_IMAGE,
        OBSERVATION_STATE,
        LANGUAGE_TOKENS,
        LANGUAGE_ATTENTION_MASK,
    }
    assert first[_BASE_IMAGE].shape == (1, 3, 4, 4)
    assert first[_WRIST_IMAGE].shape == (1, 3, 4, 4)
    assert first[OBSERVATION_STATE].shape == (1, 3)
    assert first[LANGUAGE_TOKENS].shape == (1, 6)
    assert first[LANGUAGE_ATTENTION_MASK].shape == (1, 6)
    assert first[LANGUAGE_TOKENS].dtype == torch.long
    assert first[LANGUAGE_ATTENTION_MASK].dtype == torch.bool
    assert policy.language_tokenizer.seen_texts == ["first task\n", "second task\n"]

    collated = adapter.collate((first, second))
    assert all(value.shape[0] == 2 for value in collated.values())


def test_processor_rejects_a_multi_item_tensor_at_the_single_request_boundary() -> None:
    adapter = SmolVLAAdapter()
    policy = _FakeSmolVLAPolicy()
    adapter.build_package("", policy=policy)
    request = RawRequest(
        observation={
            _BASE_IMAGE: torch.zeros(2, 3, 4, 4),
            _WRIST_IMAGE: torch.zeros(1, 3, 4, 4),
            "state": torch.zeros(3),
            "instruction_tokens": torch.zeros(6, dtype=torch.long),
        }
    )

    with pytest.raises(ModelPackageError, match="retain B=1"):
        adapter.preprocess_one(request)


@pytest.mark.smolvla
@pytest.mark.skipif(
    not (
        os.environ.get("EMBODIED_RUNTIME_SMOLVLA_CHECKPOINT")
        and os.environ.get("EMBODIED_RUNTIME_SMOLVLA_VLM_BASE")
    ),
    reason=(
        "set EMBODIED_RUNTIME_SMOLVLA_CHECKPOINT and "
        "EMBODIED_RUNTIME_SMOLVLA_VLM_BASE to local model directories"
    ),
)
def test_real_checkpoint_matches_reference_and_formal_engine() -> None:
    """Opt-in exact parity for the legacy checkpoint and staged runtime."""

    adapter = SmolVLAAdapter()
    package = adapter.build_package(
        os.environ["EMBODIED_RUNTIME_SMOLVLA_CHECKPOINT"],
        vlm_base_path=os.environ["EMBODIED_RUNTIME_SMOLVLA_VLM_BASE"],
        local_files_only=True,
    )
    assert package.spec.action_horizon == 50
    assert package.spec.action_dim == 6
    if not torch.cuda.is_available():
        return

    backend = TorchCudaBackend()
    device = next(candidate for candidate in backend.probe() if candidate.device_id == "cuda:0")
    session = backend.load(
        backend.compile(
            package,
            device,
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
        payload = adapter.synthetic_batch(batch_size=1, language_length=48, seed=17)
        cuda_payload = {key: value.to("cuda:0") for key, value in payload.items()}
        seed = 23
        noise = torch.randn(
            1,
            package.spec.action_horizon,
            int(adapter.loaded_policy.config.max_action_dim),
            generator=torch.Generator(device="cuda:0").manual_seed(seed),
            device="cuda:0",
            dtype=torch.float32,
        )

        reference_batch = adapter.loaded_policy._prepare_batch(dict(cuda_payload))
        images, image_masks = adapter.loaded_policy.prepare_images(reference_batch)
        state = adapter.loaded_policy.prepare_state(reference_batch)
        with torch.inference_mode():
            expected = adapter.loaded_policy.model.sample_actions(
                images,
                image_masks,
                reference_batch[LANGUAGE_TOKENS],
                reference_batch[LANGUAGE_ATTENTION_MASK],
                state,
                noise=noise.clone(),
            )
            expected = expected[..., : package.spec.action_dim]
            expected = adapter.loaded_policy.unnormalize_outputs({"action": expected})["action"]

            prefix = package.entrypoints[package.plan.encode](cuda_payload)
            staged = package.entrypoints[package.plan.initialize](
                {"batch_size": 1, "noise": noise.clone()}
            )
            for time_value, dt in package.plan.schedule():
                velocity = package.entrypoints[package.plan.step](
                    {"state": staged, "time": time_value, "prefix": prefix}
                )
                staged = staged + dt * velocity
            staged = package.entrypoints[package.plan.finalize]({"state": staged})["actions"]
            runtime = engine.infer(
                payload,
                num_steps=package.plan.default_num_steps,
                seed=seed,
            ).output["actions"]

        torch.testing.assert_close(staged, expected, rtol=0, atol=0)
        torch.testing.assert_close(runtime, expected[0], rtol=0, atol=0)
    finally:
        engine.close()
