from __future__ import annotations

import ast
import inspect
import math
import sys
import types
from pathlib import Path
from typing import ClassVar
from unittest import mock

import pytest

import embodied_runtime.models.vln.streamvln as streamvln_package
from embodied_runtime.models.vln.streamvln import (
    StreamVLNConfig,
    StreamVLNEvaluator,
    StreamVLNInferenceError,
    StreamVLNLoadError,
    StreamVLNNativeOutputError,
    StreamVLNRuntime,
    normalize_native_actions,
)
from embodied_runtime.models.vln.streamvln.history import compact_aligned_history


@pytest.mark.parametrize("actions", [[], [True], [1.0], [4], [1, 1, 1, 1, 1]])
def test_invalid_native_actions_are_not_coerced(actions: list[object]) -> None:
    with pytest.raises(StreamVLNNativeOutputError):
        normalize_native_actions(actions)


def test_official_text_action_parser_preserves_order() -> None:
    assert StreamVLNEvaluator.parse_actions("plan: ↑ then ←, →, STOP") == [1, 2, 3, 0]
    assert StreamVLNEvaluator.parse_actions("no native action") == []


def test_memory_rollover_retains_effective_history_and_bounds_all_aligned_lists() -> None:
    evaluator = StreamVLNEvaluator.__new__(StreamVLNEvaluator)
    evaluator.num_history = 8
    evaluator.frame_ids = list(range(32))
    evaluator.rgb_list = [("rgb", step) for step in evaluator.frame_ids]
    evaluator.depth_list = [("depth", step) for step in evaluator.frame_ids]
    evaluator.pose_list = [("pose", step) for step in evaluator.frame_ids]
    evaluator.intrinsic_list = [("intrinsic", step) for step in evaluator.frame_ids]

    evaluator._compact_history_for_rollover(32)
    assert evaluator.frame_ids == list(range(0, 32, 4))
    assert evaluator._history_count == 8

    for step in range(32, 64):
        evaluator.frame_ids.append(step)
        evaluator.rgb_list.append(("rgb", step))
        evaluator.depth_list.append(("depth", step))
        evaluator.pose_list.append(("pose", step))
        evaluator.intrinsic_list.append(("intrinsic", step))
    assert len(evaluator.rgb_list) == evaluator.num_history + 32

    evaluator._compact_history_for_rollover(64)
    assert evaluator.frame_ids == list(range(0, 64, 8))
    assert evaluator.rgb_list == [("rgb", step) for step in evaluator.frame_ids]
    assert evaluator.depth_list == [("depth", step) for step in evaluator.frame_ids]
    assert evaluator.pose_list == [("pose", step) for step in evaluator.frame_ids]
    assert evaluator.intrinsic_list == [("intrinsic", step) for step in evaluator.frame_ids]


def test_history_compaction_is_model_independent_and_keeps_payloads_aligned() -> None:
    compacted = compact_aligned_history(
        list(range(12)),
        ([f"rgb-{step}" for step in range(12)], [f"pose-{step}" for step in range(12)]),
        next_step_id=12,
        num_history=3,
    )

    assert compacted.frame_ids == [0, 4, 8]
    assert compacted.aligned == (
        ["rgb-0", "rgb-4", "rgb-8"],
        ["pose-0", "pose-4", "pose-8"],
    )


def test_checkpoint_names_preserve_hub_ids_and_materialize_existing_paths(tmp_path: Path) -> None:
    local_checkpoint = tmp_path / "checkpoint"
    local_checkpoint.mkdir()

    remote = StreamVLNConfig(streamvln_root=tmp_path, model_path="  org/model  ", warmup=False)
    local = StreamVLNConfig(
        streamvln_root=tmp_path,
        model_path=local_checkpoint,
        warmup=False,
    )

    assert remote.model_path == "org/model"
    assert local.model_path == str(local_checkpoint.resolve())


def test_streamvln_package_drops_engine_alias_and_model_domain_dependencies() -> None:
    legacy_engine_name = "Lazy" + "StreamVLN" + "Engine"
    assert legacy_engine_name not in streamvln_package.__all__
    assert not hasattr(streamvln_package, legacy_engine_name)

    package_dir = Path(inspect.getfile(streamvln_package)).parent
    forbidden = (
        "embodied_runtime.backends",
        "embodied_runtime.engine",
        "embodied_runtime.policies",
        "embodied_runtime.tasks",
    )
    for source_path in package_dir.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imported = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        imported.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            module == prefix or module.startswith(f"{prefix}.")
            for module in imported
            for prefix in forbidden
        ), source_path


class _FakeNativeModel:
    events: ClassVar[list[tuple[object, ...]]] = []

    def __init__(self) -> None:
        self.model = types.SimpleNamespace(num_history=None)

    @classmethod
    def from_pretrained(cls, model_path, **options):
        vision_tower = _FakeVisionTower()
        vision_tower.load_model(device_map="must-not-trigger-download")
        assert vision_tower.is_loaded
        assert len(vision_tower.vision_tower.vision_model.encoder.layers) == 2
        assert isinstance(vision_tower.vision_tower.vision_model.head, _FakeIdentity)
        cls.events.append(
            (
                "model",
                model_path,
                options["attn_implementation"],
                options["torch_dtype"],
            )
        )
        model = cls()
        model.loaded_vision_tower = vision_tower
        return model

    def reset(self, batch_size):
        self.events.append(("model_reset", batch_size))

    def requires_grad_(self, enabled):
        self.events.append(("requires_grad", enabled))
        return self

    def to(self, device):
        self.events.append(("model_to", device))
        return self

    def eval(self):
        self.events.append(("model_eval",))
        return self


class _LoadedFakeEvaluator:
    def __init__(self, sensor_config, **options) -> None:
        _FakeNativeModel.events.append(
            (
                "evaluator",
                options["device"],
                options["num_future_steps"],
                sensor_config["rgb_height"],
            )
        )
        self.step_id = 0

    def reset_memory(self) -> None:
        self.step_id = 0


class _FakeIdentity:
    pass


class _FakeVisionModel:
    def __init__(self, config) -> None:
        _FakeNativeModel.events.append(("embedded_vision_construct", config))
        self.vision_model = types.SimpleNamespace(
            encoder=types.SimpleNamespace(layers=["layer-0", "layer-1", "removed-layer"]),
            head="downloaded-pooling-head",
        )

    def requires_grad_(self, enabled):
        _FakeNativeModel.events.append(("embedded_vision_requires_grad", enabled))
        return self


class _FakeVisionTower:
    def __init__(self) -> None:
        self.config = object()
        self.is_loaded = False

    def load_model(self, device_map=None) -> None:
        _FakeNativeModel.events.append(("standalone_vision_download", device_map))


def _fake_heavy_modules(
    events: list[tuple[object, ...]],
    repository: Path,
) -> dict[str, types.ModuleType]:
    fake_torch = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def set_per_process_memory_fraction(fraction, device=None):
            events.append(("memory_fraction", fraction, device))

    fake_torch.cuda = FakeCuda()
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
    fake_torch.float32 = object()
    fake_torch.device = lambda value: f"device:{value}"

    fake_transformers = types.ModuleType("transformers")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_path, **options):
            events.append(("tokenizer", model_path, options["model_max_length"]))
            return object()

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_path, **options):
            events.append(("config", model_path, options["local_files_only"]))
            return object()

    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoConfig = FakeAutoConfig

    fake_model_package = types.ModuleType("model")
    fake_model_package.__path__ = []
    fake_native_module = types.ModuleType("model.stream_video_vln")
    fake_native_module.__file__ = str(repository / "streamvln/model/stream_video_vln.py")
    fake_native_module.StreamVLNForCausalLM = _FakeNativeModel

    fake_llava = types.ModuleType("llava")
    fake_llava.__path__ = []
    fake_llava.__file__ = str(repository / "llava/__init__.py")
    fake_llava_model = types.ModuleType("llava.model")
    fake_llava_model.__path__ = []
    fake_multimodal = types.ModuleType("llava.model.multimodal_encoder")
    fake_multimodal.__path__ = []
    fake_siglip = types.ModuleType("llava.model.multimodal_encoder.siglip_encoder")
    fake_siglip.__file__ = str(repository / "llava/model/multimodal_encoder/siglip_encoder.py")
    fake_siglip.SigLipVisionTower = _FakeVisionTower
    fake_siglip.SigLipVisionModel = _FakeVisionModel
    fake_siglip.nn = types.SimpleNamespace(Identity=_FakeIdentity)
    fake_utils = types.ModuleType("utils")
    fake_utils.__path__ = []
    fake_utils_module = types.ModuleType("utils.utils")
    fake_utils_module.__file__ = str(repository / "streamvln/utils/utils.py")
    return {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "model": fake_model_package,
        "model.stream_video_vln": fake_native_module,
        "llava": fake_llava,
        "llava.model": fake_llava_model,
        "llava.model.multimodal_encoder": fake_multimodal,
        "llava.model.multimodal_encoder.siglip_encoder": fake_siglip,
        "utils": fake_utils,
        "utils.utils": fake_utils_module,
    }


def test_load_is_lazy_adds_both_official_paths_and_limits_cuda_before_weights(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "StreamVLN"
    (repository / "streamvln").mkdir(parents=True)
    events: list[tuple[object, ...]] = []
    _FakeNativeModel.events = events
    runtime = StreamVLNRuntime(
        streamvln_root=repository,
        model_path="fixture-checkpoint",
        cuda_memory_fraction=0.45,
        local_files_only=True,
        warmup=False,
        evaluator_factory=_LoadedFakeEvaluator,
    )

    assert not runtime.loaded
    assert events == []
    modules = _fake_heavy_modules(events, repository)
    with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "path", list(sys.path)):
        runtime.load()
        runtime.load()
        assert sys.path[0] == str(repository / "streamvln")
        assert sys.path[1] == str(repository)

    assert runtime.loaded
    assert events[0] == ("memory_fraction", 0.45, "device:cuda:0")
    assert not any(event[0] == "standalone_vision_download" for event in events)
    assert [event[0] for event in events].count("embedded_vision_construct") == 1
    assert ("embedded_vision_requires_grad", False) in events
    assert [event[0] for event in events].count("model") == 1
    assert [event[0] for event in events].index("memory_fraction") < [
        event[0] for event in events
    ].index("model")
    assert ("evaluator", "device:cuda:0", 4, 1.25) in events
    # The offline construction hook is scoped to parent-checkpoint loading.
    assert _FakeVisionTower.load_model.__name__ == "load_model"


def test_cached_generic_upstream_module_from_another_repo_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "StreamVLN"
    repository.mkdir()
    foreign_llava = types.ModuleType("llava")
    foreign_llava.__file__ = str(tmp_path / "another-model/llava/__init__.py")
    runtime = StreamVLNRuntime(
        streamvln_root=repository,
        model_path="fixture-checkpoint",
        warmup=False,
    )

    with (
        mock.patch.dict(sys.modules, {"llava": foreign_llava}),
        mock.patch.object(sys, "path", list(sys.path)),
        pytest.raises(StreamVLNLoadError, match="outside"),
    ):
        runtime.load()


class _CadenceEvaluator:
    def __init__(self) -> None:
        self.step_id = 0
        self.calls: list[tuple[int, object, str, bool]] = []

    def step(self, environment_id, image, instruction, *, run_model=False):
        self.calls.append((self.step_id, image, instruction, run_model))
        if run_model:
            return [1, 2, 1, 0], 0.125, "↑←↑STOP"
        return None, 0.0, None

    def reset_memory(self) -> None:
        self.step_id = 0


class _FailingCadenceEvaluator(_CadenceEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True
        self.fail_reset = False
        self.reset_calls = 0

    def step(self, environment_id, image, instruction, *, run_model=False):
        if self.fail and self.step_id == 2:
            raise RuntimeError("synthetic generation failure")
        return super().step(
            environment_id,
            image,
            instruction,
            run_model=run_model,
        )

    def reset_memory(self) -> None:
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("synthetic reset failure")
        super().reset_memory()


def test_predict_preserves_image_instruction_actions_and_official_four_step_cadence(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "StreamVLN"
    repository.mkdir()
    runtime = StreamVLNRuntime(
        streamvln_root=repository,
        model_path="fixture-checkpoint",
        warmup=False,
    )
    evaluator = _CadenceEvaluator()
    runtime._evaluator = evaluator
    image = object()

    prediction = runtime.predict(image, "walk through the doorway, then stop")

    assert [
        (step, instruction, run_model) for step, _, instruction, run_model in evaluator.calls
    ] == [
        (0, "walk through the doorway, then stop", True),
        (1, "walk through the doorway, then stop", False),
        (2, "walk through the doorway, then stop", False),
        (3, "walk through the doorway, then stop", False),
    ]
    assert all(call_image is image for _, call_image, _, _ in evaluator.calls)
    assert prediction.actions == (1, 2, 1, 0)
    assert prediction.raw_output == "↑←↑STOP"
    assert prediction.generation_time_s == pytest.approx(0.125)
    assert prediction.as_dict() == {
        "actions": [1, 2, 1, 0],
        "raw_output": "↑←↑STOP",
        "generation_time_s": pytest.approx(0.125),
    }


def test_partial_cadence_is_discarded_and_requires_explicit_reset(tmp_path: Path) -> None:
    runtime = StreamVLNRuntime(
        streamvln_root=tmp_path,
        model_path="fixture-checkpoint",
        warmup=False,
    )
    evaluator = _FailingCadenceEvaluator()
    runtime._evaluator = evaluator

    with pytest.raises(StreamVLNInferenceError, match="requires reset"):
        runtime.predict(object(), "walk to the doorway")

    assert runtime.requires_reset
    assert evaluator.reset_calls == 1
    assert evaluator.step_id == 0
    calls_after_failure = len(evaluator.calls)
    with pytest.raises(StreamVLNInferenceError, match=r"call reset\(\)"):
        runtime.predict(object(), "walk to the doorway")
    assert len(evaluator.calls) == calls_after_failure

    evaluator.fail = False
    runtime.reset()
    prediction = runtime.predict(object(), "walk to the doorway")
    assert not runtime.requires_reset
    assert prediction.actions == (1, 2, 1, 0)


def test_failed_defensive_reset_keeps_runtime_poisoned_until_reset_succeeds(
    tmp_path: Path,
) -> None:
    runtime = StreamVLNRuntime(
        streamvln_root=tmp_path,
        model_path="fixture-checkpoint",
        warmup=False,
    )
    evaluator = _FailingCadenceEvaluator()
    evaluator.fail_reset = True
    runtime._evaluator = evaluator

    with pytest.raises(StreamVLNInferenceError, match="reset also failed"):
        runtime.predict(object(), "walk to the doorway")
    assert runtime.requires_reset

    evaluator.fail = False
    evaluator.fail_reset = False
    runtime.reset()
    assert not runtime.requires_reset


@pytest.mark.parametrize(
    "fraction",
    [True, "0.5", 0.0, -0.1, 1.01, math.nan, math.inf],
)
def test_cuda_memory_fraction_is_strictly_validated(tmp_path: Path, fraction: object) -> None:
    with pytest.raises(ValueError, match="cuda_memory_fraction"):
        StreamVLNRuntime(
            streamvln_root=tmp_path,
            cuda_memory_fraction=fraction,
            warmup=False,
        )


def test_runtime_rejects_dtype_that_evaluator_cannot_honor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only dtype='bfloat16'"):
        StreamVLNRuntime(
            streamvln_root=tmp_path,
            dtype="float16",
            warmup=False,
        )
