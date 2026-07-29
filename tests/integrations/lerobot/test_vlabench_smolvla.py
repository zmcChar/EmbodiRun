from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from embodied_runtime.integrations.lerobot import (
    VLABENCH_CAMERA_RENAME_MAP,
    LeRobotBindings,
    LeRobotRunnerError,
    VLABenchSmolVLARunner,
)


@dataclass
class _Feature:
    shape: tuple[int, ...]


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], values: list[float] | None = None) -> None:
        self.shape = shape
        self.values = values or [0.0] * (shape[-1] if shape else 0)

    def __getitem__(self, index: int) -> _FakeTensor:
        assert index == 0
        return _FakeTensor(self.shape[1:], self.values)

    def detach(self) -> _FakeTensor:
        return self

    def to(self, device: str) -> _FakeTensor:
        assert device == "cpu"
        return self

    def numpy(self) -> _FakeTensor:
        return self

    def numel(self) -> int:
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product


class _FakeConfig:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self) -> None:
        self.input_features = {
            "observation.state": _Feature((6,)),
            "observation.images.camera1": _Feature((3, 256, 256)),
            "observation.images.camera2": _Feature((3, 256, 256)),
            "observation.images.camera3": _Feature((3, 256, 256)),
        }
        self.output_features = {"action": _Feature((7,))}
        self.device = "old-device"
        self.use_amp = False
        self.chunk_size = 50
        self.n_action_steps = 50
        self.pretrained_path: str | None = None
        self.pretrained_revision: str | None = None

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs: Any) -> _FakeConfig:
        cls.calls.append((checkpoint, kwargs))
        return cls()


class _FakeEnvConfig:
    def __init__(self, task: str) -> None:
        self.task = task
        self.features = {
            "action": _Feature((7,)),
            "agent_pos": _Feature((7,)),
            "pixels/image": _Feature((480, 480, 3)),
            "pixels/second_image": _Feature((480, 480, 3)),
            "pixels/wrist_image": _Feature((480, 480, 3)),
        }


class _FakePipeline:
    def __init__(self, name: str, events: list[str], *, stats: bool = False) -> None:
        self.name = name
        self.events = events
        self.stats = stats

    def __call__(self, value: Any) -> Any:
        self.events.append(self.name)
        return value

    def state_dict(self) -> dict[str, dict[str, _FakeTensor]]:
        if not self.stats:
            return {}
        return {"normalizer": {"observation.state.mean": _FakeTensor((6,))}}


class _FakePolicy:
    def __init__(self, config: _FakeConfig, events: list[str]) -> None:
        self.config = config
        self.events = events
        self.eval_calls = 0
        self.reset_calls = 0
        self.selected_batches: list[dict[str, Any]] = []
        self.next_shape = (1, 7)
        self._queues: dict[str, deque[_FakeTensor]] = {"action": deque()}

    def eval(self) -> None:
        self.eval_calls += 1

    def reset(self) -> None:
        self.reset_calls += 1
        self._queues["action"].clear()
        self.events.append("reset")

    def select_action(self, batch: dict[str, Any]) -> _FakeTensor:
        self.events.append("policy")
        self.selected_batches.append(batch)
        if not self._queues["action"]:
            self._queues["action"].extend(
                _FakeTensor(self.next_shape, list(range(7))) for _ in range(3)
            )
        return self._queues["action"].popleft()

    def parameters(self) -> Iterable[_FakeTensor]:
        return [_FakeTensor((2, 3)), _FakeTensor((4,))]


@dataclass
class _Fixture:
    bindings: LeRobotBindings
    calls: dict[str, Any]
    events: list[str]


@pytest.fixture
def fake_runtime() -> _Fixture:
    _FakeConfig.calls.clear()
    events: list[str] = []
    calls: dict[str, Any] = {"make_policy": 0, "make_processors": 0}

    def make_policy(**kwargs: Any) -> _FakePolicy:
        calls["make_policy"] += 1
        calls["policy_kwargs"] = kwargs
        policy = _FakePolicy(kwargs["cfg"], events)
        calls["policy"] = policy
        return policy

    def make_processors(**kwargs: Any) -> tuple[_FakePipeline, _FakePipeline]:
        calls["make_processors"] += 1
        calls["processor_kwargs"] = kwargs
        return (
            _FakePipeline("pre", events, stats=True),
            _FakePipeline("post", events, stats=True),
        )

    def make_env_processors(**kwargs: Any) -> tuple[_FakePipeline, _FakePipeline]:
        calls["env_processor_kwargs"] = kwargs
        return _FakePipeline("env_pre", events), _FakePipeline("env_post", events)

    def preprocess(observation: dict[str, Any]) -> dict[str, Any]:
        events.append("raw_preprocess")
        return {"converted": observation}

    bindings = LeRobotBindings(
        pre_trained_config=_FakeConfig,
        vlabench_env_config=_FakeEnvConfig,
        make_policy=make_policy,
        make_pre_post_processors=make_processors,
        make_env_pre_post_processors=make_env_processors,
        preprocess_observation=preprocess,
        inference_context=lambda device, amp: nullcontext(),
        version="0.6.0",
    )
    return _Fixture(bindings=bindings, calls=calls, events=events)


def _raw_observation() -> dict[str, Any]:
    return {
        "pixels": {
            "image": object(),
            "second_image": object(),
            "wrist_image": object(),
        },
        "agent_pos": object(),
    }


def test_package_import_keeps_heavy_native_dependencies_lazy() -> None:
    source_root = Path(__file__).parents[3] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    script = """
import sys
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
from embodied_runtime.integrations.lerobot import VLABenchSmolVLARunner
assert VLABenchSmolVLARunner
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
assert not any(name == "lerobot" or name.startswith("lerobot.") for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=source_root.parent,
        env=environment,
    )


def test_load_mirrors_native_lerobot_factories_once(fake_runtime: _Fixture) -> None:
    runner = VLABenchSmolVLARunner(
        checkpoint="checkpoint",
        task="select_fruit",
        device="cuda:0",
        revision="revision",
        backbone_path="/models/smolvlm2",
        local_files_only=True,
        bindings=fake_runtime.bindings,
    )

    assert runner.load() is runner
    assert runner.load() is runner
    assert fake_runtime.calls["make_policy"] == 1
    assert fake_runtime.calls["make_processors"] == 1
    assert _FakeConfig.calls == [
        (
            "checkpoint",
            {"revision": "revision", "local_files_only": True},
        )
    ]

    policy_kwargs = fake_runtime.calls["policy_kwargs"]
    assert policy_kwargs["env_cfg"].task == "select_fruit"
    assert policy_kwargs["rename_map"] == VLABENCH_CAMERA_RENAME_MAP
    assert policy_kwargs["cfg"].pretrained_path == "checkpoint"
    assert policy_kwargs["cfg"].pretrained_revision == "revision"
    assert policy_kwargs["cfg"].device == "cuda:0"
    assert policy_kwargs["cfg"].vlm_model_name == "/models/smolvlm2"

    overrides = fake_runtime.calls["processor_kwargs"]["preprocessor_overrides"]
    assert overrides == {
        "device_processor": {"device": "cuda:0"},
        "rename_observations_processor": {
            "rename_map": VLABENCH_CAMERA_RENAME_MAP,
        },
        "tokenizer_processor": {
            "tokenizer_name": "/models/smolvlm2",
        },
    }
    assert runner.facts.input_feature_shapes["observation.state"] == (6,)
    assert runner.facts.environment_feature_shapes["agent_pos"] == (7,)
    assert runner.facts.output_feature_shapes["action"] == (7,)
    assert runner.facts.preprocessor_stat_shapes["normalizer.observation.state.mean"] == (6,)
    assert runner.facts.model_parameter_count == 10


def test_select_action_uses_official_pipeline_order_and_measures_policy(
    fake_runtime: _Fixture,
) -> None:
    clock = iter((10.0, 10.25))
    runner = VLABenchSmolVLARunner(
        bindings=fake_runtime.bindings,
        clock=lambda: next(clock),
    )

    result = runner.select_action(_raw_observation(), "pick the red fruit")

    assert result.action.shape == (7,)
    assert result.inference_latency_s == pytest.approx(0.25)
    assert result.generated_chunk
    assert result.queue_remaining == 2
    assert fake_runtime.events == [
        "raw_preprocess",
        "env_pre",
        "pre",
        "policy",
        "post",
        "env_post",
    ]
    policy = fake_runtime.calls["policy"]
    assert policy.selected_batches[0]["task"] == ["pick the red fruit"]


def test_cuda_synchronization_wraps_measured_policy_call(fake_runtime: _Fixture) -> None:
    events = fake_runtime.events
    bindings = replace(
        fake_runtime.bindings,
        synchronize=lambda device: events.append(f"sync:{device}"),
    )
    runner = VLABenchSmolVLARunner(
        bindings=bindings,
        device="cuda:0",
        clock=iter((10.0, 10.25)).__next__,
    )

    runner.select_action(_raw_observation(), "pick the red fruit")

    assert events == [
        "raw_preprocess",
        "env_pre",
        "pre",
        "sync:cuda:0",
        "policy",
        "sync:cuda:0",
        "post",
        "env_post",
    ]


def test_changed_prompt_and_explicit_boundary_reset_cached_action_chunk(
    fake_runtime: _Fixture,
) -> None:
    runner = VLABenchSmolVLARunner(
        bindings=fake_runtime.bindings,
        clock=lambda: 0.0,
    )

    runner.select_action(_raw_observation(), "first subgoal")
    runner.select_action(_raw_observation(), "second subgoal")
    runner.select_action(
        _raw_observation(),
        "second subgoal",
        reset_action_queue=True,
    )

    policy = fake_runtime.calls["policy"]
    assert policy.reset_calls == 2
    assert fake_runtime.events.count("reset") == 2


def test_reset_action_queue_is_public_and_clears_active_prompt(
    fake_runtime: _Fixture,
) -> None:
    runner = VLABenchSmolVLARunner(
        bindings=fake_runtime.bindings,
        clock=lambda: 0.0,
    )
    runner.select_action(_raw_observation(), "subgoal")
    runner.reset_action_queue()
    runner.select_action(_raw_observation(), "different prompt")

    policy = fake_runtime.calls["policy"]
    assert policy.reset_calls == 1


@pytest.mark.parametrize("shape", [(2, 7), (1, 6), (7, 1)])
def test_rejects_anything_other_than_one_7d_action(
    fake_runtime: _Fixture,
    shape: tuple[int, ...],
) -> None:
    runner = VLABenchSmolVLARunner(
        bindings=fake_runtime.bindings,
        clock=lambda: 0.0,
    )
    runner.load()
    fake_runtime.calls["policy"].next_shape = shape

    with pytest.raises(LeRobotRunnerError, match="7-D action"):
        runner.select_action(_raw_observation(), "task")
