from types import SimpleNamespace

import pytest

from rlinf_deploy.bindings.lerobot.so101.pi05.runner import (
    SO101Pi05Spec,
    execute,
)
from rlinf_deploy.robots.sensors.cameras import CameraFrame, V4L2CameraConfig


def spec(*, max_steps=2):
    return SO101Pi05Spec(
        runtime_id="so101-runtime",
        prompt="pick up the block",
        model_endpoint="http://127.0.0.1:8000",
        robot_port="/dev/ttyACM0",
        robot_id="so101-1",
        calibration_id="calibrated-arm",
        calibration_dir="/calibration",
        disable_torque_on_disconnect=True,
        max_joint_step_deg=12.0,
        max_gripper_step=20.0,
        step_limit_mode="reject",
        cameras=(
            V4L2CameraConfig(
                "observation.images.front",
                "/dev/video0",
                640,
                480,
                30.0,
            ),
        ),
        max_steps=max_steps,
        control_hz=5.0,
        request_timeout_s=60.0,
    )


class FakeCameras:
    def __init__(self):
        self.capture_calls = 0
        self.closed = False

    def capture(self):
        self.capture_calls += 1
        return (CameraFrame("observation.images.front", "image/jpeg", b"jpeg"),)

    def close(self):
        self.closed = True


class FakeRobot:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, endpoint, *, timeout_s):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def health(self):
        return {"status": "ok"}


class FakeRuntime:
    def __init__(self, robot, client, *, instruction):
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.step_calls = []
        self.closed = False

    def step(self, images):
        self.step_calls.append(images)
        return SimpleNamespace(session_revision=len(self.step_calls))

    def close(self):
        self.closed = True


def test_binding_runner_executes_bounded_steps_and_releases_hardware() -> None:
    cameras = FakeCameras()
    robot = FakeRobot()
    runtimes = []
    events = []
    sleeps = []
    clock = iter((0.0, 0.1, 0.2, 0.3))

    def runtime_factory(*args, **kwargs):
        runtime = FakeRuntime(*args, **kwargs)
        runtimes.append(runtime)
        return runtime

    execute(
        spec(),
        camera_factory=lambda _specs: cameras,
        robot_factory=lambda config: robot,
        client_factory=FakeClient,
        runtime_factory=runtime_factory,
        emit=events.append,
        monotonic=lambda: next(clock),
        sleep=sleeps.append,
    )

    assert cameras.capture_calls == 2
    assert runtimes[0].instruction == "pick up the block"
    assert len(runtimes[0].step_calls) == 2
    assert sleeps == pytest.approx([0.1])
    assert events[-1] == {
        "event": "complete",
        "runtime": "so101-runtime",
        "steps": 2,
    }
    assert runtimes[0].closed is True
    assert robot.closed is True
    assert cameras.closed is True


def test_binding_runner_releases_hardware_when_inference_fails() -> None:
    cameras = FakeCameras()
    robot = FakeRobot()

    class FailingRuntime(FakeRuntime):
        def step(self, images):
            raise RuntimeError("inference failed")

    runtime = None

    def runtime_factory(*args, **kwargs):
        nonlocal runtime
        runtime = FailingRuntime(*args, **kwargs)
        return runtime

    with pytest.raises(RuntimeError, match="inference failed"):
        execute(
            spec(max_steps=1),
            camera_factory=lambda _specs: cameras,
            robot_factory=lambda config: robot,
            client_factory=FakeClient,
            runtime_factory=runtime_factory,
        )

    assert runtime is not None and runtime.closed is True
    assert robot.closed is True
    assert cameras.closed is True
