from __future__ import annotations

import threading
import time
from typing import Any

from embodied_runtime.robots.unitree.go2.agent.control import (
    ActionExecutor,
    RobotState,
    RobotTransport,
)


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
