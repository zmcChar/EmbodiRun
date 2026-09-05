import time

import pytest

from rlinf_deploy.robots import RobotAction
from rlinf_deploy.robots.lerobot.so101 import (
    SO101_POSITION_FEATURES,
    SO101Adapter,
    SO101AdapterError,
    SO101Config,
)


class FakeSO101:
    def __init__(self, positions):
        self.positions = dict(zip(SO101_POSITION_FEATURES, positions))
        self.is_connected = False
        self.is_calibrated = True
        self.actions = []

    def connect(self, *, calibrate):
        assert calibrate is False
        self.is_connected = True

    def get_observation(self):
        return dict(self.positions)

    def send_action(self, action):
        self.actions.append(dict(action))
        self.positions.update(action)

    def disconnect(self):
        self.is_connected = False


def action(joints, gripper):
    return RobotAction(
        timestamp_s=time.time(),
        values={
            "type": "joint_position",
            "joint_positions_deg": joints,
            "gripper_position": gripper,
        },
    )


def test_so101_reject_mode_remains_fail_closed() -> None:
    hardware = FakeSO101([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    adapter = SO101Adapter(
        SO101Config(port="/dev/fake", max_joint_step_deg=5.0),
        controller=hardware,
    )
    adapter.connect()

    with pytest.raises(SO101AdapterError, match="exceeds 5.000000 degrees"):
        adapter.execute(action([20.0, 0.0, 0.0, 0.0, 0.0], 50.0))

    assert hardware.actions == []


def test_so101_clip_mode_bounds_every_joint_and_gripper() -> None:
    hardware = FakeSO101([0.0, 10.0, -10.0, 20.0, -20.0, 50.0])
    adapter = SO101Adapter(
        SO101Config(
            port="/dev/fake",
            max_joint_step_deg=5.0,
            max_gripper_step=10.0,
            step_limit_mode="clip",
        ),
        controller=hardware,
    )
    adapter.connect()

    adapter.execute(action([20.0, -20.0, -7.0, 16.0, -25.0], 80.0))

    assert hardware.actions == [
        {
            "shoulder_pan.pos": 5.0,
            "shoulder_lift.pos": 5.0,
            "elbow_flex.pos": -7.0,
            "wrist_flex.pos": 16.0,
            "wrist_roll.pos": -25.0,
            "gripper.pos": 60.0,
        }
    ]


def test_so101_clip_mode_bounds_each_command_in_a_sequence() -> None:
    hardware = FakeSO101([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    adapter = SO101Adapter(
        SO101Config(
            port="/dev/fake",
            max_joint_step_deg=5.0,
            max_gripper_step=10.0,
            step_limit_mode="clip",
        ),
        controller=hardware,
    )
    adapter.connect()

    target = action([20.0, 0.0, 0.0, 0.0, 0.0], 50.0)
    adapter.execute(target)
    adapter.execute(target)

    assert [item["shoulder_pan.pos"] for item in hardware.actions] == [5.0, 10.0]


def test_so101_config_rejects_unknown_step_limit_mode() -> None:
    with pytest.raises(ValueError, match="step_limit_mode"):
        SO101Config(port="/dev/fake", step_limit_mode="unsafe")
