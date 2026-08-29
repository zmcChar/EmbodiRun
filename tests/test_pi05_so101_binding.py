from types import SimpleNamespace

import pytest

from rlinf_deploy.bindings.lerobot.so101.pi05 import (
    Pi05SO101ActionMapper,
    Pi05SO101ActionMapperError,
    Pi05SO101Runtime,
)
from rlinf_deploy.inference import ImagePayload, PolicyAction, PolicyResult, Session
from rlinf_deploy.robots.lerobot.so101 import SO101_POSITION_FEATURES

DEFAULT_ROW = (1, 2, 3, 4, 5, 6)


def result(*, names=SO101_POSITION_FEATURES, rows=(DEFAULT_ROW,)) -> PolicyResult:
    return PolicyResult(
        request_id="r1",
        session_id="s1",
        step_id=0,
        session_revision=1,
        action_space="pi05.action_chunk.v1",
        actions=(
            PolicyAction(
                "action_chunk",
                {
                    "data": [list(row) for row in rows],
                    "feature_names": list(names),
                },
            ),
        ),
    )


def test_mapper_uses_feature_names_instead_of_wire_order() -> None:
    mapper = Pi05SO101ActionMapper()
    names = tuple(reversed(SO101_POSITION_FEATURES))
    action = mapper.map_result(result(names=names, rows=((60, 50, 40, 30, 20, 10),)))
    assert action.values == {
        "type": "joint_position",
        "joint_positions_deg": [10.0, 20.0, 30.0, 40.0, 50.0],
        "gripper_position": 60.0,
    }


def test_mapper_rejects_missing_and_duplicate_feature_names() -> None:
    mapper = Pi05SO101ActionMapper()
    with pytest.raises(Pi05SO101ActionMapperError, match="do not match SO-101"):
        mapper.map_result(result(names=tuple(f"other-{index}" for index in range(6))))
    with pytest.raises(Pi05SO101ActionMapperError, match="unique"):
        mapper.map_result(result(names=("duplicate",) * 6))


def test_mapper_rejects_unconsumed_action_rows() -> None:
    with pytest.raises(Pi05SO101ActionMapperError, match="exactly one action row"):
        Pi05SO101ActionMapper().map_result(result(rows=(DEFAULT_ROW, DEFAULT_ROW)))


class FakeRobot:
    robot_id = "so101-test"

    def __init__(self) -> None:
        self.executed = []
        self.stop_count = 0

    def observe(self):
        return SimpleNamespace(
            timestamp_s=1.0,
            values={"joint_positions_deg": [0.0] * 5, "gripper_position": 0.0},
        )

    def execute(self, action) -> None:
        self.executed.append(action)

    def stop(self) -> None:
        self.stop_count += 1


class FakeClient:
    def __init__(self) -> None:
        self.observations = []
        self.closed = False

    def open_session(self, *, robot_id, action_space, metadata=None):
        assert robot_id == "so101-test"
        assert action_space == "pi05.action_chunk.v1"
        return Session("s1", 0)

    def step(self, observation):
        self.observations.append(observation)
        return result()

    def reset(self, session_id, *, request_id):
        return Session(session_id, 1)

    def close(self, session_id):
        self.closed = True


def test_runtime_reuses_generic_session_loop() -> None:
    robot = FakeRobot()
    client = FakeClient()
    runtime = Pi05SO101Runtime(robot, client, instruction="pick")
    runtime.step((ImagePayload("camera-0", "image/jpeg", b"123"),))
    assert len(robot.executed) == 1
    assert client.observations[0].state["joint_positions_deg"] == [0.0] * 5
    runtime.close()
    assert robot.stop_count == 1
    assert client.closed
