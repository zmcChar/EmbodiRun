from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import embodied_runtime.integrations.navigation.vvla.activevln as activevln_module
from embodied_runtime.integrations.navigation.vvla import (
    VvlaActiveVLNNavigationPolicy,
    VvlaActiveVLNRuntime,
    activevln_actions_to_waypoint_plan,
    validate_activevln_action_text,
)
from embodied_runtime.policies.navigation.errors import NavigationPolicyError
from embodied_runtime.tasks.navigation import (
    EncodedRGBFrame,
    NavigationObservation,
    NavigationRequest,
)


def _action(name: str, value: int | None):
    return SimpleNamespace(name=name, value=value)


def _request(sequence: int, *, episode: str = "episode-1", reset: bool = False):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), (30, 60, 90)).save(stream, format="JPEG")
    observation = NavigationObservation(
        episode_id=episode,
        sequence=sequence,
        reset=reset,
        rgb_frames=(
            EncodedRGBFrame(
                sequence=sequence,
                captured_at_s=float(sequence + 1),
                data=stream.getvalue(),
                width=3,
                height=2,
            ),
        ),
    )
    return NavigationRequest("walk through the doorway", observation)


class _Runtime:
    def __init__(self) -> None:
        self.loads = 0
        self.resets = 0
        self.cancels = 0
        self.closed = 0
        self.inputs = []
        self.actions = (_action("turn left", 30), _action("move forward", 50))

    def load(self) -> None:
        self.loads += 1

    def reset(self) -> None:
        self.resets += 1

    def cancel(self) -> None:
        self.cancels += 1

    def close(self) -> None:
        self.closed += 1

    def predict(self, rgb, instruction: str, *, episode_id: str):
        self.inputs.append((rgb, instruction, episode_id))
        return SimpleNamespace(actions=self.actions)


def test_activevln_mapping_preserves_native_magnitudes_and_stop() -> None:
    plan = activevln_actions_to_waypoint_plan(
        (
            _action("turn left", 30),
            _action("move forward", 50),
            _action("stop", None),
        ),
        observation_sequence=9,
    )

    assert plan.observation_sequence == 9
    assert plan.terminal is True
    assert len(plan.waypoints) == 2
    assert plan.waypoints[0].yaw_rad == pytest.approx(0.5235987756)
    assert plan.waypoints[1].x_m == pytest.approx(0.5 * 3**0.5 / 2)
    assert plan.waypoints[1].y_m == pytest.approx(0.25)


@pytest.mark.parametrize(
    "actions",
    [
        (),
        (_action("move forward", 100),),
        (_action("turn left", 90),),
        (_action("dance", None),),
        (_action("stop", None), _action("move forward", 25)),
        tuple(_action("move forward", 25) for _ in range(4)),
    ],
)
def test_activevln_mapping_rejects_output_outside_pinned_grammar(actions) -> None:
    with pytest.raises(ValueError):
        activevln_actions_to_waypoint_plan(actions, observation_sequence=1)


def test_activevln_mapping_rejects_numeric_values_with_the_wrong_type() -> None:
    with pytest.raises(ValueError, match="unsupported ActiveVLN action"):
        activevln_actions_to_waypoint_plan(
            (_action("move forward", 25.0),),  # type: ignore[arg-type]
            observation_sequence=1,
        )


def test_activevln_text_validation_requires_canonical_output() -> None:
    actions = (_action("move forward", 25), _action("turn right", 30))
    validate_activevln_action_text("move forward 25cm, turn right 30 degrees", actions)

    for unsafe in (
        "move forward",
        "turn left",
        "please do not move forward 25cm",
        "move forward 25cm,, stop",
    ):
        with pytest.raises(ValueError, match="non-canonical"):
            validate_activevln_action_text(unsafe, actions)

    with pytest.raises(ValueError, match="final action"):
        validate_activevln_action_text(
            "stop, move forward 25cm",
            (_action("stop", None), _action("move forward", 25)),
        )


def test_activevln_text_validation_checks_the_typed_parse() -> None:
    with pytest.raises(ValueError, match="do not match"):
        validate_activevln_action_text(
            "move forward 25cm",
            (_action("turn left", 15),),
        )


def test_activevln_policy_is_lazy_and_resets_recurrent_episode() -> None:
    runtime = _Runtime()
    policy = VvlaActiveVLNNavigationPolicy(runtime=runtime)  # type: ignore[arg-type]

    assert runtime.loads == 0
    asyncio.run(policy.prepare())
    first = asyncio.run(policy.plan(_request(1, reset=True)))
    second = asyncio.run(policy.plan(_request(2)))
    asyncio.run(policy.aclose())

    assert runtime.loads == 1
    assert runtime.resets == 1
    assert runtime.closed == 1
    assert len(first.waypoints) == len(second.waypoints) == 2
    assert runtime.inputs[0][0].shape == (2, 3, 3)
    assert runtime.inputs[0][1:] == ("walk through the doorway", "episode-1")


def test_activevln_policy_fails_closed_and_discards_bad_session() -> None:
    runtime = _Runtime()
    runtime.actions = (_action("move forward", 100),)
    policy = VvlaActiveVLNNavigationPolicy(runtime=runtime)  # type: ignore[arg-type]

    asyncio.run(policy.prepare())
    with pytest.raises(NavigationPolicyError, match="unsupported ActiveVLN action"):
        asyncio.run(policy.plan(_request(1, reset=True)))

    assert runtime.resets == 2  # initial reset and cleanup after invalid output
    runtime.actions = (_action("stop", None),)
    recovered = asyncio.run(policy.plan(_request(2)))
    assert recovered.terminal is True
    assert not recovered.waypoints


def test_cancelled_prepare_drains_load_before_releasing_runtime() -> None:
    load_started = threading.Event()
    allow_load = threading.Event()
    events: list[str] = []

    class BlockingRuntime(_Runtime):
        def load(self) -> None:
            events.append("load-start")
            load_started.set()
            assert allow_load.wait(timeout=2.0)
            events.append("load-end")

        def close(self) -> None:
            events.append("close")

    async def scenario() -> None:
        policy = VvlaActiveVLNNavigationPolicy(runtime=BlockingRuntime())  # type: ignore[arg-type]
        task = asyncio.create_task(policy.prepare())
        assert await asyncio.to_thread(load_started.wait, 1.0)
        task.cancel()
        allow_load.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert events == ["load-start", "load-end", "close"]


def test_cancelled_close_stays_closed_and_drains_runtime() -> None:
    close_started = threading.Event()
    allow_close = threading.Event()
    events: list[str] = []

    class BlockingRuntime(_Runtime):
        def close(self) -> None:
            events.append("close-start")
            close_started.set()
            assert allow_close.wait(timeout=2.0)
            events.append("close-end")

    async def scenario() -> None:
        policy = VvlaActiveVLNNavigationPolicy(runtime=BlockingRuntime())  # type: ignore[arg-type]
        task = asyncio.create_task(policy.aclose())
        assert await asyncio.to_thread(close_started.wait, 1.0)
        task.cancel()
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RuntimeError, match="closed"):
            await policy.prepare()

    asyncio.run(scenario())
    assert events == ["close-start", "close-end"]


def test_runtime_builds_safe_recurrent_engine_and_uses_explicit_sessions(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, value=None) -> None:
            self.value = value
            self.operations: list[object] = []

        def permute(self, *dimensions: int):
            self.operations.append(("permute", dimensions))
            return self

        def float(self):
            self.operations.append("float")
            return self

        def div_(self, value: float):
            self.operations.append(("div", value))
            return self

    class FakeCuda:
        emptied = False

        @classmethod
        def is_available(cls) -> bool:
            return True

        @classmethod
        def empty_cache(cls) -> None:
            cls.emptied = True

    class FakeTorch:
        float16 = object()
        bfloat16 = object()
        float32 = object()
        long = object()
        cuda = FakeCuda

        @staticmethod
        def from_numpy(value):
            tensor = FakeTensor(value)
            calls["pixels"] = tensor
            return tensor

        @staticmethod
        def empty(*shape, dtype=None):
            tensor = FakeTensor()
            tensor.operations.append(("empty", shape, dtype))
            return tensor

    class FakePolicy:
        def __init__(self) -> None:
            self.cast_to = None

        def to(self, dtype):
            self.cast_to = dtype
            return self

    class FakeEngineConfig:
        def __init__(self, **options) -> None:
            self.options = options
            calls["engine_config"] = self

    @dataclass(frozen=True)
    class FakeSessionKey:
        env_id: str
        episode_id: str
        rollout_id: int = 0

    class FakeObservation:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    class FakeBackend:
        def __init__(self) -> None:
            self.generated = []
            self.resets = []
            self.cancels = []

        def generate(self, observations, *, session_ids):
            self.generated.append((observations, session_ids))
            text = "move forward 25cm, turn left 15 degrees"
            parsed = SimpleNamespace(
                valid=True,
                raw_text=text,
                actions=(_action("move forward", 25), _action("turn left", 15)),
            )
            trace = SimpleNamespace(
                parsed_actions=parsed,
                text=text,
                token_ids=SimpleNamespace(tolist=lambda: [11, 12]),
            )
            return [SimpleNamespace(trace=trace, latency_ms=7.5)]

        def reset_sessions(self, keys) -> None:
            self.resets.append(keys)

        def cancel_sessions(self, keys) -> None:
            self.cancels.append(keys)

    class FakeVvla:
        def __init__(self, policy, *, engine_config) -> None:
            calls["vvla_policy"] = policy
            calls["vvla_config"] = engine_config
            self.core = SimpleNamespace(dtype="torch.bfloat16")
            self.backend = FakeBackend()

    fake_policy = FakePolicy()

    def make_policy(name: str, **options):
        calls["make_policy"] = (name, options)
        return fake_policy

    modules = {
        "torch": FakeTorch,
        "vvla": SimpleNamespace(EngineConfig=FakeEngineConfig, Vvla=FakeVvla),
        "vvla.policies": SimpleNamespace(make_policy=make_policy),
        "vvla.types": SimpleNamespace(
            Observation=FakeObservation,
            SessionKey=FakeSessionKey,
        ),
    }
    monkeypatch.setattr(
        activevln_module.importlib,
        "import_module",
        lambda name: modules[name],
    )

    runtime = VvlaActiveVLNRuntime(
        checkpoint="Arvil/Qwen2.5-VL-3B_rl_r2r_4000",
        dtype="bfloat16",
        attention="sdpa",
        max_new_tokens=24,
        max_context=4096,
        env_id="test-env",
    )
    monkeypatch.setattr(runtime, "_make_importable", lambda: None)
    runtime.load()

    assert fake_policy.cast_to is FakeTorch.bfloat16
    assert calls["vvla_policy"] is fake_policy
    config = calls["engine_config"]
    assert isinstance(config, FakeEngineConfig)
    assert config.options == {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "use_cuda_graph": False,
        "capture_full_loop": False,
    }
    name, options = calls["make_policy"]
    assert name == "activevln"
    assert options["attention"] == "sdpa"
    assert options["max_new_tokens"] == 24
    assert options["max_context"] == 4096

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    prediction = runtime.predict(rgb, "walk through the doorway", episode_id="episode-1")
    pixels = calls["pixels"]
    assert isinstance(pixels, FakeTensor)
    assert pixels.value is rgb
    assert pixels.operations == [("permute", (2, 0, 1)), "float", ("div", 255.0)]
    assert prediction.text == "move forward 25cm, turn left 15 degrees"
    assert prediction.token_ids == (11, 12)

    first_key = FakeSessionKey("test-env", "episode-1")
    backend = runtime.backend
    observation = backend.generated[0][0][0]
    assert observation.instruction == "walk through the doorway"
    assert backend.generated[0][1] == [first_key]

    runtime.predict(rgb, "walk through the doorway", episode_id="episode-2")
    second_key = FakeSessionKey("test-env", "episode-2")
    assert backend.resets == [[first_key]]
    runtime.cancel()
    assert backend.cancels == [[second_key]]
    runtime.reset()
    assert backend.resets[-1] == [second_key]
    runtime.close()
    assert FakeCuda.emptied is True


def test_vvla_runtime_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pinned immutable"):
        VvlaActiveVLNRuntime(revision="main")


def test_vvla_cancel_before_worker_start_is_not_lost() -> None:
    runtime = VvlaActiveVLNRuntime()
    runtime._engine = object()
    runtime._observation_type = object()
    runtime.cancel()

    with pytest.raises(NavigationPolicyError, match="cancelled"):
        runtime.predict(
            np.zeros((4, 5, 3), dtype=np.uint8),
            "Walk to the door.",
            episode_id="episode-1",
        )

    runtime.reset()
    assert not runtime._cancelled.is_set()
