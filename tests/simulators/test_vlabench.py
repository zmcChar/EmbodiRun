from __future__ import annotations

import sys

import pytest

from embodied_runtime.contracts import RobotAction
from embodied_runtime.simulators.vlabench import (
    VLABenchSimulatorEndpoint,
    observation_fingerprint,
)

np = pytest.importorskip("numpy")


class _Task:
    def get_instruction(self):
        return "Put the red toy into the left container."


class _Inner:
    task = _Task()


class _FakeEnvironment:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._env = _Inner()
        self.task_description = "cluster_toy"
        self.closed = False

    def reset(self, seed=None, **kwargs):
        del kwargs
        return {
            "pixels": {
                "image": np.full((2, 2, 3), seed or 0, dtype=np.uint8),
                "second_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            "agent_pos": np.zeros(7, dtype=np.float64),
        }, {"seed": seed}

    def step(self, action):
        return self.reset(seed=1)[0], 1.0, True, False, {"is_success": True}

    def close(self):
        self.closed = True


def test_vlabench_module_keeps_heavy_dependencies_lazy():
    assert "lerobot.envs.vlabench" not in sys.modules


def test_endpoint_normalizes_instruction_action_and_success():
    environment = _FakeEnvironment()

    def environment_factory(**kwargs):
        environment.kwargs.update(kwargs)
        return environment

    endpoint = VLABenchSimulatorEndpoint(
        "cluster_toy",
        max_episode_steps=10,
        render_resolution=(2, 2),
        environment_factory=environment_factory,
        clock=lambda: 4.0,
    )

    observation = endpoint.reset(seed=7)

    assert endpoint.instruction == "Put the red toy into the left container."
    assert observation.metadata["instruction"] == endpoint.instruction
    assert observation.metadata["initial_fingerprint"] == endpoint.initial_fingerprint
    assert endpoint.raw_environment is environment

    outcome = endpoint.step(
        RobotAction(timestamp_s=4.0, values={"action": np.zeros(7, dtype=np.float32)})
    )

    assert outcome.success is True
    assert outcome.done is True
    assert outcome.observation.metadata["step_index"] == 1
    endpoint.close()
    assert environment.closed


def test_initial_fingerprint_is_content_stable():
    first = {"b": np.array([1, 2]), "a": {"x": np.array([[3]], dtype=np.int16)}}
    second = {"a": {"x": np.array([[3]], dtype=np.int16)}, "b": np.array([1, 2])}

    assert observation_fingerprint(first) == observation_fingerprint(second)
    second["b"][0] = 9
    assert observation_fingerprint(first) != observation_fingerprint(second)


def test_endpoint_rejects_wrong_action_shape():
    endpoint = VLABenchSimulatorEndpoint(
        "cluster_toy",
        render_resolution=(2, 2),
        environment_factory=_FakeEnvironment,
    )
    endpoint.reset(seed=1)

    with pytest.raises(ValueError, match="shape"):
        endpoint.step(RobotAction(timestamp_s=0.0, values=np.zeros(6)))
