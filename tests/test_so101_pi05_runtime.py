from types import SimpleNamespace

from rlinf_deploy.bindings.lerobot.so101.pi05.runtime import Pi05SO101Runtime
from rlinf_deploy.inference import ImagePayload, Session
from rlinf_deploy.robots import RobotAction
from rlinf_deploy.robots.sensors.cameras import CameraFrame


class FakeRobot:
    robot_id = "so101-1"

    def __init__(self) -> None:
        self.actions = []

    def observe(self):
        return SimpleNamespace(timestamp_s=1.5, values={"shoulder_pan.pos": 0.0})

    def execute(self, action):
        self.actions.append(action)

    def stop(self):
        return None


class FakeClient:
    def __init__(self) -> None:
        self.observation = None

    def open_session(self, *, robot_id, action_space):
        assert robot_id == "so101-1"
        assert action_space == "pi05.action_chunk.v1"
        return Session("session-1", 0)

    def step(self, observation):
        self.observation = observation
        return SimpleNamespace(session_revision=1)


class FakeMapper:
    def map_result(self, _result):
        return RobotAction(2.0, {"type": "joint_position"})


def test_so101_binding_converts_camera_frames_to_inference_images() -> None:
    robot = FakeRobot()
    client = FakeClient()
    runtime = Pi05SO101Runtime(
        robot,
        client,
        instruction="pick up the block",
        mapper=FakeMapper(),
    )

    runtime.step((CameraFrame("observation.images.front", "image/jpeg", b"jpeg"),))

    assert client.observation is not None
    assert client.observation.images == (
        ImagePayload("observation.images.front", "image/jpeg", b"jpeg"),
    )
    assert client.observation.metadata == {"robot_timestamp_s": 1.5}
    assert len(robot.actions) == 1
