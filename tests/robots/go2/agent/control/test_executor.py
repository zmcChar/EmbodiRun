from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from embodied_runtime.robots.go2.agent.control import (
    ActionExecutor,
    ApiError,
    DryRunTransport,
    RobotState,
    RobotTransport,
)


class TiltedTransport(DryRunTransport):
    def state(self) -> RobotState:
        return replace(super().state(), roll=0.8)


class InvalidStateTransport(DryRunTransport):
    def state(self) -> RobotState:
        return replace(super().state(), yaw=float("nan"))


class PausedBeforeMoveTransport(RobotTransport):
    """Pause a worker after state validation but before its first Move."""

    name = "paused-before-move"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._worker_state_calls = 0
        self.before_move = threading.Event()
        self.release_worker = threading.Event()
        self.commands: list[tuple[Any, ...]] = []

    def state(self) -> RobotState:
        should_pause = False
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            if threading.current_thread().name.startswith("go2-"):
                self._worker_state_calls += 1
                should_pause = self._worker_state_calls == 2
        if should_pause:
            self.before_move.set()
            self.release_worker.wait(timeout=2.0)
        return RobotState(
            received_at=time.monotonic(),
            sequence=sequence,
            position=(0.0, 0.0, 0.0),
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            velocity=(0.0, 0.0, 0.0),
            yaw_rate=0.0,
        )

    def move(self, vx: float, vy: float, yaw_rate: float) -> int:
        with self._lock:
            self.commands.append(("move", vx, vy, yaw_rate))
        return 0

    def stop(self) -> int:
        with self._lock:
            self.commands.append(("stop",))
        return 0

    def posture(self, action: str) -> int:
        with self._lock:
            self.commands.append(("posture", action))
        return 0


@pytest.fixture
def executor() -> ActionExecutor:
    value = ActionExecutor(DryRunTransport(), operator_motion_ready=True)
    yield value
    value.close()


def wait_for_result(executor: ActionExecutor, timeout: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = executor.snapshot()
        if snapshot["active_action"] is None and snapshot["last_result"]:
            return snapshot["last_result"]
        time.sleep(0.02)
    pytest.fail("action did not finish")


def test_timed_move_stops(executor: ActionExecutor) -> None:
    status, response = executor.execute(
        {
            "action": "move",
            "vx": 0.2,
            "vy": 0.0,
            "yaw_rate": 0.0,
            "duration_s": 0.1,
        }
    )

    assert status == 202
    assert response["accepted"]
    assert wait_for_result(executor)["status"] == "completed"
    assert executor.snapshot()["robot_state"]["velocity"][0] == pytest.approx(0.0)


def test_timed_move_velocity_can_be_updated_without_new_lease(
    executor: ActionExecutor,
) -> None:
    status, response = executor.execute(
        {
            "action": "move",
            "vx": 0.2,
            "vy": 0.0,
            "yaw_rate": 0.0,
            "duration_s": 0.5,
        }
    )
    action_id = response["action_id"]
    update_status, update = executor.execute(
        {
            "action": "update_move",
            "action_id": action_id,
            "vx": 0.15,
            "vy": 0.0,
            "yaw_rate": 0.2,
        }
    )

    assert status == 202
    assert update_status == 200
    assert update["action_id"] == action_id
    active = executor.snapshot()["active_action"]
    assert active["id"] == action_id
    assert active["parameters"]["vx"] == 0.15
    assert active["parameters"]["yaw_rate"] == 0.2
    assert wait_for_result(executor)["status"] == "completed"


def test_stream_move_expires_without_heartbeat(
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "embodied_runtime.robots.go2.agent.control.executor.STREAM_HEARTBEAT_TIMEOUT_S",
        0.1,
    )
    executor.execute(
        {
            "action": "stream_move",
            "vx": 0.1,
            "vy": 0.0,
            "yaw_rate": 0.0,
            "duration_s": 0.8,
        }
    )

    result = wait_for_result(executor)
    assert result["status"] == "failed"
    assert result["detail"] == "stream heartbeat expired"


def test_update_move_rejects_wrong_action_id(executor: ActionExecutor) -> None:
    executor.execute(
        {
            "action": "move",
            "vx": 0.2,
            "vy": 0.0,
            "yaw_rate": 0.0,
            "duration_s": 0.2,
        }
    )
    with pytest.raises(ApiError, match="action_id"):
        executor.execute(
            {
                "action": "update_move",
                "action_id": "not-active",
                "vx": 0.0,
                "vy": 0.0,
                "yaw_rate": 0.2,
            }
        )


def test_distance_is_closed_loop(executor: ActionExecutor) -> None:
    executor.execute({"action": "move_distance", "distance_m": 0.05, "speed_mps": 0.2})

    result = wait_for_result(executor)
    assert result["status"] == "completed"
    assert result["measured_distance_m"] >= 0.045
    assert executor.snapshot()["robot_state"]["position"][0] >= 0.045


def test_turn_is_closed_loop(executor: ActionExecutor) -> None:
    executor.execute({"action": "turn", "angle_rad": 0.1, "yaw_rate_rps": 0.5})

    result = wait_for_result(executor)
    assert result["status"] == "completed"
    assert result["measured_angle_rad"] >= 0.09
    assert abs(executor.snapshot()["robot_state"]["yaw"]) >= 0.09


@pytest.mark.parametrize(
    "payload",
    [
        {
            "action": "move",
            "vx": 2.0,
            "vy": 0.0,
            "yaw_rate": 0.0,
            "duration_s": 1.0,
        },
        {"action": "turn", "angle_rad": math.pi, "yaw_rate_rps": 0.5},
    ],
)
def test_rejects_out_of_range_motion(
    executor: ActionExecutor,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ApiError):
        executor.execute(payload)


def test_rejects_motion_when_robot_is_tilted() -> None:
    executor = ActionExecutor(TiltedTransport(), operator_motion_ready=True)
    try:
        with pytest.raises(ApiError, match="tilt"):
            executor.execute({"action": "move_distance", "distance_m": 0.05, "speed_mps": 0.1})
    finally:
        executor.close()


def test_rejects_non_finite_robot_state() -> None:
    executor = ActionExecutor(InvalidStateTransport(), operator_motion_ready=True)
    try:
        with pytest.raises(ApiError, match="invalid numeric"):
            executor.execute({"action": "turn", "angle_rad": 0.1, "yaw_rate_rps": 0.2})
    finally:
        executor.close()


def test_motion_requires_explicit_operator_interlock() -> None:
    executor = ActionExecutor(DryRunTransport())
    try:
        assert not executor.snapshot()["operator_motion_ready"]
        with pytest.raises(ApiError, match="operator motion-ready"):
            executor.execute({"action": "turn", "angle_rad": 0.1, "yaw_rate_rps": 0.2})
    finally:
        executor.close()


def test_snapshot_preserves_state_sequence_and_received_at(
    executor: ActionExecutor,
) -> None:
    first = executor.snapshot()["robot_state"]
    second = executor.snapshot()["robot_state"]

    assert "received_at" in first
    assert first["received_at_unix"] > 1_000_000_000
    assert "sequence" in first
    assert second["sequence"] > first["sequence"]


def test_explicit_stop_disarms_operator_interlock(executor: ActionExecutor) -> None:
    result = executor.stop("emergency_test")

    assert result["stopped"]
    assert not result["operator_motion_ready"]
    assert not executor.snapshot()["operator_motion_ready"]
    with pytest.raises(ApiError, match="operator motion-ready"):
        executor.execute({"action": "turn", "angle_rad": 0.1, "yaw_rate_rps": 0.2})


def test_close_is_idempotent_and_rejects_new_motion(executor: ActionExecutor) -> None:
    executor.close()
    executor.close()

    with pytest.raises(ApiError, match="executor is closing"):
        executor.execute({"action": "turn", "angle_rad": 0.1, "yaw_rate_rps": 0.2})


def test_stop_cannot_be_followed_by_worker_move() -> None:
    transport = PausedBeforeMoveTransport()
    executor = ActionExecutor(transport, operator_motion_ready=True)
    stop_result: dict[str, Any] = {}
    try:
        executor.execute(
            {
                "action": "move",
                "vx": 0.2,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "duration_s": 1.0,
            }
        )
        assert transport.before_move.wait(timeout=1.0)

        stop_thread = threading.Thread(
            target=lambda: stop_result.update(executor.stop("race_test"))
        )
        stop_thread.start()
        deadline = time.monotonic() + 1.0
        while not executor._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert executor._stop_event.is_set()
        transport.release_worker.set()
        stop_thread.join(timeout=2.0)

        assert not stop_thread.is_alive()
        assert stop_result["worker_joined"]
        assert stop_result["stopped"]
        assert not any(command[0] == "move" for command in transport.commands)
        assert sum(command[0] == "stop" for command in transport.commands) >= 2
    finally:
        transport.release_worker.set()
        executor.close()


def test_posture_waits_for_motion_cancellation() -> None:
    transport = PausedBeforeMoveTransport()
    executor = ActionExecutor(transport, operator_motion_ready=True)
    posture_result: dict[str, Any] = {}
    posture_errors: list[Exception] = []
    try:
        executor.execute(
            {
                "action": "move",
                "vx": 0.2,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "duration_s": 1.0,
            }
        )
        assert transport.before_move.wait(timeout=1.0)

        def change_posture() -> None:
            try:
                status, payload = executor.execute({"action": "stand_down"})
                posture_result.update(status=status, payload=payload)
            except Exception as exc:  # noqa: BLE001 - capture cross-thread assertion result
                posture_errors.append(exc)

        posture_thread = threading.Thread(target=change_posture)
        posture_thread.start()
        deadline = time.monotonic() + 1.0
        while not executor._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert executor._stop_event.is_set()
        assert not any(command[0] == "posture" for command in transport.commands)

        transport.release_worker.set()
        posture_thread.join(timeout=2.0)
        assert not posture_thread.is_alive()
        assert posture_errors == []
        assert posture_result["status"] == 200
        assert not posture_result["payload"]["operator_motion_ready"]
        assert not executor.snapshot()["operator_motion_ready"]
        posture_index = next(
            index for index, command in enumerate(transport.commands) if command[0] == "posture"
        )
        assert any(command[0] == "stop" for command in transport.commands[:posture_index])
        assert not any(command[0] == "move" for command in transport.commands)
    finally:
        transport.release_worker.set()
        executor.close()
