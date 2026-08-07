from __future__ import annotations

from typing import Any

import pytest

from embodied_runtime.robots.action import RobotAction
from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.simulators import (
    EpisodeStep,
    GymLikeSimulatorAdapter,
    SimulatorCapabilities,
    SimulatorEndpoint,
)


def _observation_mapper(raw: dict[str, Any]) -> RobotObservation:
    return RobotObservation(timestamp_s=float(raw["tick"]), values=raw)


def _action_mapper(action: RobotAction) -> Any:
    return action.values["command"]


class FiveValueEnvironment:
    def __init__(self) -> None:
        self.reset_kwargs: dict[str, Any] = {}
        self.actions: list[Any] = []
        self.closed = False

    def reset(self, **kwargs: Any) -> tuple[dict[str, int], dict[str, str]]:
        self.reset_kwargs = kwargs
        return {"tick": 0}, {"reset_source": "fake"}

    def observe(self) -> dict[str, int]:
        return {"tick": 1}

    def step(self, action: Any) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return {"tick": 2}, 1.5, True, False, {"won": 1, "progress": 0.75}

    def close(self) -> None:
        self.closed = True


def _capabilities(**changes: Any) -> SimulatorCapabilities:
    values = {
        "name": "fake",
        "environment_id": "FakeTask-v0",
        "embodiment": "test-arm",
        "action_space_id": "joint-position-v1",
    }
    values.update(changes)
    return SimulatorCapabilities(**values)


def test_capabilities_validate_and_normalize_identifiers_and_features() -> None:
    capabilities = _capabilities(
        name=" fake ",
        features=frozenset({" disturbances ", "replay"}),
    )

    assert capabilities.name == "fake"
    assert capabilities.features == frozenset({"disturbances", "replay"})
    with pytest.raises(ValueError, match="name"):
        _capabilities(name=" ")
    with pytest.raises(ValueError, match="features"):
        _capabilities(features=frozenset({""}))
    with pytest.raises(TypeError, match="features"):
        _capabilities(features="replay")
    with pytest.raises(TypeError, match="supports_seed"):
        _capabilities(supports_seed=1)


def test_five_value_adapter_maps_data_and_implements_runtime_protocol() -> None:
    environment = FiveValueEnvironment()
    adapter = GymLikeSimulatorAdapter(
        environment,
        _capabilities(),
        observation_mapper=_observation_mapper,
        action_mapper=_action_mapper,
        success_mapper=lambda info: bool(info["won"]),
        subgoal_mapper=lambda info: float(info["progress"]),
    )

    assert isinstance(adapter, SimulatorEndpoint)
    reset_observation = adapter.reset("pick-and-place", seed=7, options={"difficulty": "hard"})
    observed = adapter.observe()
    outcome = adapter.step(RobotAction(timestamp_s=1.0, values={"command": [0.1, 0.2]}))

    assert reset_observation.values == {"tick": 0}
    assert adapter.last_reset_info == {"reset_source": "fake"}
    assert environment.reset_kwargs == {
        "seed": 7,
        "options": {"difficulty": "hard", "task": "pick-and-place"},
    }
    assert observed.values == {"tick": 1}
    assert environment.actions == [[0.1, 0.2]]
    assert outcome == EpisodeStep(
        observation=RobotObservation(timestamp_s=2.0, values={"tick": 2}),
        reward=1.5,
        terminated=True,
        truncated=False,
        success=True,
        subgoal_progress=0.75,
        info={"won": 1, "progress": 0.75},
    )
    assert outcome.done

    adapter.close()
    adapter.close()
    assert environment.closed
    with pytest.raises(RuntimeError, match="closed"):
        adapter.observe()


@pytest.mark.parametrize(
    ("done", "info", "expected_terminated", "expected_truncated"),
    [
        (True, {}, True, False),
        (True, {"TimeLimit.truncated": True}, False, True),
        (False, {}, False, False),
    ],
)
def test_four_value_adapter_normalizes_legacy_done(
    done: bool,
    info: dict[str, Any],
    expected_terminated: bool,
    expected_truncated: bool,
) -> None:
    class FourValueEnvironment:
        def reset(self) -> RobotObservation:
            return RobotObservation(timestamp_s=0.0, values={"tick": 0})

        def step(self, action: RobotAction) -> tuple[RobotObservation, float, bool, dict[str, Any]]:
            return RobotObservation(timestamp_s=1.0, values={"tick": 1}), 0.0, done, info

    adapter = GymLikeSimulatorAdapter(FourValueEnvironment(), _capabilities())
    reset_observation = adapter.reset()
    outcome = adapter.step(RobotAction(timestamp_s=0.5, values=[0.0]))

    assert adapter.observe() is outcome.observation
    assert reset_observation.values == {"tick": 0}
    assert outcome.terminated is expected_terminated
    assert outcome.truncated is expected_truncated


def test_adapter_rejects_invalid_results_and_seed_capability() -> None:
    class InvalidEnvironment:
        def reset(self) -> dict[str, int]:
            return {"tick": 0}

        def step(self, action: Any) -> tuple[int, int, int]:
            return 1, 2, 3

    adapter = GymLikeSimulatorAdapter(
        InvalidEnvironment(),
        _capabilities(supports_seed=False),
        observation_mapper=_observation_mapper,
    )

    with pytest.raises(ValueError, match="seeded"):
        adapter.reset(seed=1)
    adapter.reset()
    with pytest.raises(ValueError, match="four or five"):
        adapter.step(RobotAction(timestamp_s=0.0, values=[0.0]))


def test_episode_step_validates_reward_and_progress() -> None:
    observation = RobotObservation(timestamp_s=0.0, values={})

    with pytest.raises(ValueError, match="finite"):
        EpisodeStep(observation, float("nan"), False, False)
    with pytest.raises(ValueError, match="between zero and one"):
        EpisodeStep(observation, 0.0, False, False, subgoal_progress=1.1)
