from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import embodied_runtime.integrations.navigation.activevln_transformers as module
from embodied_runtime.integrations.navigation.activevln_transformers import (
    TransformersActiveVLNRuntime,
)
from embodied_runtime.policies.navigation.errors import NavigationPolicyError


class _InputTensor:
    def __init__(self, shape) -> None:
        self.shape = shape
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class _TokenRow:
    def __init__(self, values) -> None:
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class _Generated:
    def __init__(self, values) -> None:
        self.values = list(values)

    def __getitem__(self, index):
        assert index == 0
        return _TokenRow(self.values)


class _Sequences:
    def __init__(self, values) -> None:
        self.values = list(values)

    def __getitem__(self, index):
        assert isinstance(index, tuple)
        return _Generated(self.values)


def _fake_modules():
    calls: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def synchronize(device=None) -> None:
            calls.setdefault("synchronizes", []).append(device)

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            calls["emptied"] = True

    class FakeTorch:
        float16 = "torch.float16"
        bfloat16 = "torch.bfloat16"
        float32 = "torch.float32"
        cuda = FakeCuda

        @staticmethod
        def inference_mode():
            return nullcontext()

    class FakeModel:
        config = SimpleNamespace(
            model_type="qwen2_5_vl",
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            text_config=SimpleNamespace(rope_scaling={"mrope_section": [16, 24, 24]}),
        )

        def __init__(self) -> None:
            self.device = None
            self.evaluating = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluating = True
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype="torch.bfloat16")])

        def generate(self, **options):
            calls.setdefault("generate", []).append(options)
            return _Sequences([11, 12, 13])

    model = FakeModel()

    class FakeModelClass:
        @staticmethod
        def from_pretrained(checkpoint, **options):
            calls["model_load"] = (checkpoint, options)
            return model

    class FakeTokenizer:
        def __init__(self) -> None:
            self.responses = [
                "move forward 25cm, turn right 30 degrees",
                "stop",
                "turn left 15 degrees",
            ]

        def decode(self, tokens, **options):
            calls.setdefault("decode", []).append((tokens.tolist(), options))
            return self.responses.pop(0)

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def apply_chat_template(self, conversation, **options):
            calls.setdefault("conversations", []).append(list(conversation))
            calls.setdefault("template_options", []).append(options)
            return "serialized prompt"

        def __call__(self, **options):
            calls.setdefault("processor", []).append(options)
            return {
                "input_ids": _InputTensor((1, 100)),
                "attention_mask": _InputTensor((1, 100)),
                "pixel_values": _InputTensor((384, 1176)),
                "image_grid_thw": _InputTensor((1, 3)),
            }

    processor = FakeProcessor()

    class FakeProcessorClass:
        @staticmethod
        def from_pretrained(checkpoint, **options):
            calls["processor_load"] = (checkpoint, options)
            return processor

    class FakeStoppingCriteriaList(list):
        pass

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeProcessorClass,
        Qwen2_5_VLForConditionalGeneration=FakeModelClass,
        StoppingCriteriaList=FakeStoppingCriteriaList,
    )
    return calls, FakeTorch, fake_transformers


def _runtime(monkeypatch):
    calls, fake_torch, fake_transformers = _fake_modules()
    real_import = module.importlib.import_module

    def fake_import(name: str):
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    monkeypatch.setattr(module, "verify_activevln_checkpoint", lambda *args: None)
    runtime = TransformersActiveVLNRuntime(
        checkpoint="Arvil/Qwen2.5-VL-3B_rl_r2r_4000",
        dtype="bfloat16",
        attention="eager",
        max_new_tokens=64,
    )
    return runtime, calls


def test_transformers_runtime_loads_bf16_and_uses_full_history(monkeypatch) -> None:
    runtime, calls = _runtime(monkeypatch)
    runtime.load()

    _, model_options = calls["model_load"]
    assert model_options["torch_dtype"] == "torch.bfloat16"
    assert model_options["attn_implementation"] == "eager"
    assert model_options["local_files_only"] is True
    _, processor_options = calls["processor_load"]
    assert processor_options["use_fast"] is False
    assert runtime.engine_dtype == "bfloat16"

    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    first = runtime.predict(rgb, "Walk to the door.", episode_id="episode-1")
    second = runtime.predict(rgb, "Walk to the door.", episode_id="episode-1")

    assert [(item.name, item.value) for item in first.actions] == [
        ("move forward", 25),
        ("turn right", 30),
    ]
    assert [(item.name, item.value) for item in second.actions] == [("stop", None)]
    conversations = calls["conversations"]
    assert [item["role"] for item in conversations[0]] == ["system", "user"]
    assert [item["role"] for item in conversations[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    processor_calls = calls["processor"]
    assert len(processor_calls[0]["images"]) == 1
    assert len(processor_calls[1]["images"]) == 2
    for generated in calls["generate"]:
        assert generated["use_cache"] is True
        assert generated["do_sample"] is False
        assert generated["repetition_penalty"] == 1.05
        assert generated["temperature"] is None

    runtime.reset()
    runtime.predict(rgb, "Walk to the door.", episode_id="episode-2")
    assert [item["role"] for item in calls["conversations"][-1]] == ["system", "user"]
    runtime.close()
    assert calls["emptied"] is True


def test_transformers_runtime_does_not_commit_an_invalid_response(monkeypatch) -> None:
    runtime, calls = _runtime(monkeypatch)
    runtime.load()
    runtime._processor.tokenizer.responses[0] = "please do not move forward 25cm"

    with pytest.raises(NavigationPolicyError, match="non-canonical"):
        runtime.predict(
            np.zeros((4, 5, 3), dtype=np.uint8),
            "Walk to the door.",
            episode_id="episode-1",
        )

    assert runtime._active_episode_id is None
    assert runtime._conversation == []
    assert runtime._images == []
    assert len(calls["conversations"]) == 1


def test_transformers_runtime_rejects_vvla_only_attention() -> None:
    with pytest.raises(ValueError, match="eager_bc"):
        TransformersActiveVLNRuntime(attention="eager_bc")


def test_transformers_runtime_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pinned immutable"):
        TransformersActiveVLNRuntime(revision="main")


def test_transformers_cancel_before_worker_start_is_not_lost(monkeypatch) -> None:
    runtime, calls = _runtime(monkeypatch)
    runtime.load()
    runtime.cancel()

    with pytest.raises(NavigationPolicyError, match="cancelled"):
        runtime.predict(
            np.zeros((4, 5, 3), dtype=np.uint8),
            "Walk to the door.",
            episode_id="episode-1",
        )

    assert "generate" not in calls
    runtime.reset()
    runtime.predict(
        np.zeros((4, 5, 3), dtype=np.uint8),
        "Walk to the door.",
        episode_id="episode-1",
    )
    assert len(calls["generate"]) == 1
