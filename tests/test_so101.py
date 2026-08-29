import pytest

from rlinf_deploy.robots import RobotAction
from rlinf_deploy.robots.lerobot.so101 import (
    SO101_ACTION_SPACE,
    SO101_POSITION_FEATURES,
    SO101Adapter,
    SO101AdapterError,
    SO101Config,
)


class FakeSO101:
    def __init__(self, *, calibrated: bool = True, connect_error=None) -> None:
        self.connected_with = None
        self.disconnected = False
        self.is_calibrated = calibrated
        self.is_connected = False
        self.connect_error = connect_error
        self.positions = dict.fromkeys(SO101_POSITION_FEATURES, 0.0)
        self.sent = []

    def connect(self, *, calibrate: bool) -> None:
        self.connected_with = calibrate
        self.is_connected = True
        if self.connect_error is not None:
            raise self.connect_error

    def get_observation(self):
        return dict(self.positions)

    def send_action(self, action):
        self.sent.append(dict(action))
        self.positions.update(action)
        return dict(action)

    def disconnect(self) -> None:
        self.is_connected = False
        self.disconnected = True


def action(values):
    return RobotAction(
        timestamp_s=1.0,
        values=values,
        metadata={"action_space": SO101_ACTION_SPACE},
    )


def test_observe_and_execute_follow_lerobot_contract() -> None:
    device = FakeSO101()
    adapter = SO101Adapter(SO101Config(port="/dev/ttyACM0"), lerobot_robot=device)
    assert device.connected_with is False
    assert adapter.observe().values == {
        "joint_positions_deg": [0.0] * 5,
        "gripper_position": 0.0,
    }

    adapter.execute(
        action(
            {
                "type": "joint_position",
                "joint_positions_deg": [1.0, 2.0, 3.0, 4.0, 5.0],
                "gripper_position": 10.0,
            }
        )
    )
    assert device.sent[-1] == dict(
        zip(SO101_POSITION_FEATURES, [1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    )


def test_step_limits_and_gripper_range_are_fail_closed() -> None:
    device = FakeSO101()
    adapter = SO101Adapter(
        SO101Config(port="/dev/ttyACM0", max_joint_step_deg=5.0),
        lerobot_robot=device,
    )
    with pytest.raises(SO101AdapterError, match="joint step"):
        adapter.execute(
            action(
                {
                    "type": "joint_position",
                    "joint_positions_deg": [6.0, 0.0, 0.0, 0.0, 0.0],
                    "gripper_position": 0.0,
                }
            )
        )
    with pytest.raises(SO101AdapterError, match=r"\[0, 100\]"):
        adapter.execute(
            action(
                {
                    "type": "joint_position",
                    "joint_positions_deg": [0.0] * 5,
                    "gripper_position": 101.0,
                }
            )
        )
    assert device.sent == []


def test_stop_holds_position_and_close_disconnects() -> None:
    device = FakeSO101()
    device.positions = dict(zip(SO101_POSITION_FEATURES, range(6)))
    adapter = SO101Adapter(SO101Config(port="/dev/ttyACM0"), lerobot_robot=device)
    adapter.stop()
    assert device.sent == [device.positions]
    adapter.close()
    assert device.disconnected


def test_uncalibrated_robot_is_disconnected_before_use() -> None:
    device = FakeSO101(calibrated=False)
    with pytest.raises(SO101AdapterError, match="lerobot-calibrate"):
        SO101Adapter(SO101Config(port="/dev/ttyACM0"), lerobot_robot=device)
    assert device.disconnected


def test_partial_connection_is_disconnected_before_error_propagates() -> None:
    failure = RuntimeError("bus failure")
    device = FakeSO101(connect_error=failure)
    with pytest.raises(RuntimeError, match="bus failure") as caught:
        SO101Adapter(SO101Config(port="/dev/ttyACM0"), lerobot_robot=device)
    assert caught.value is failure
    assert device.disconnected


def test_wrong_action_space_and_unknown_fields_are_rejected() -> None:
    device = FakeSO101()
    adapter = SO101Adapter(SO101Config(port="/dev/ttyACM0"), lerobot_robot=device)
    wrong_space = RobotAction(
        timestamp_s=1.0,
        values={"type": "stop"},
        metadata={"action_space": "franka.fr3.control.v1"},
    )
    with pytest.raises(SO101AdapterError, match="unsupported action space"):
        adapter.execute(wrong_space)
    with pytest.raises(SO101AdapterError, match="fields do not match"):
        adapter.execute(
            action(
                {
                    "type": "joint_position",
                    "joint_positions_deg": [0.0] * 5,
                    "gripper_position": 0.0,
                    "surprise": 1,
                }
            )
        )
