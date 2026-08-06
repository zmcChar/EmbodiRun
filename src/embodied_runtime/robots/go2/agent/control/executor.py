"""Bounded action admission, execution, heartbeat, and stop semantics."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import asdict
from http import HTTPStatus
from typing import Any, ClassVar

from .config import (
    CONTROL_PERIOD_S,
    MAX_LINEAR_SPEED_MPS,
    MAX_MOTION_DURATION_S,
    MAX_MOVE_DISTANCE_M,
    MAX_TILT_RAD,
    MAX_TURN_ANGLE_RAD,
    MAX_YAW_RATE_RPS,
    STATE_MAX_AGE_S,
    STREAM_HEARTBEAT_TIMEOUT_S,
    WORKER_STOP_JOIN_TIMEOUT_S,
)
from .transport import RobotTransport
from .types import ApiError, RobotState, normalize_angle, require_number


class ActionExecutor:
    """Validate API actions and execute at most one bounded motion at a time."""

    POSTURE_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {"stand_up", "stand_down", "balance_stand", "recovery_stand"}
    )

    def __init__(
        self,
        transport: RobotTransport,
        *,
        operator_motion_ready: bool = False,
    ) -> None:
        self.transport = transport
        # This is an explicit operator interlock. Do not infer readiness from
        # undocumented numeric gait or mode values.
        self.operator_motion_ready = bool(operator_motion_ready)
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._active: dict[str, Any] | None = None
        self._last_result: dict[str, Any] | None = None
        self._closing = False
        self._closed = False

    def snapshot(self) -> dict[str, Any]:
        state = self.transport.state()
        with self._lock:
            active = None
            if self._active:
                active = dict(self._active)
                parameters = active.get("parameters")
                if isinstance(parameters, dict):
                    active["parameters"] = dict(parameters)
            result = dict(self._last_result) if self._last_result else None
            operator_motion_ready = self.operator_motion_ready
            closing = self._closing
        state_json = None
        state_age_s = None
        if state is not None:
            state_age_s = max(0.0, time.monotonic() - state.received_at)
            state_json = asdict(state)
        return {
            "transport": self.transport.name,
            "operator_motion_ready": operator_motion_ready,
            "closing": closing,
            "robot_state_available": state is not None,
            "robot_state_fresh": state_age_s is not None and state_age_s <= STATE_MAX_AGE_S,
            "robot_state_age_s": round(state_age_s, 3) if state_age_s is not None else None,
            "robot_state": state_json,
            "active_action": active,
            "last_result": result,
            "limits": {
                "linear_speed_mps": MAX_LINEAR_SPEED_MPS,
                "yaw_rate_rps": MAX_YAW_RATE_RPS,
                "move_distance_m": MAX_MOVE_DISTANCE_M,
                "turn_angle_rad": MAX_TURN_ANGLE_RAD,
                "motion_duration_s": MAX_MOTION_DURATION_S,
                "tilt_rad": MAX_TILT_RAD,
            },
        }

    def execute(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        action = payload.get("action")
        if not isinstance(action, str):
            raise ApiError(HTTPStatus.BAD_REQUEST, "action must be a string")

        if action == "stop":
            return HTTPStatus.OK, self.stop("api_stop")

        if action in self.POSTURE_ACTIONS:
            return HTTPStatus.OK, self._execute_posture(action)

        if action in {"move", "stream_move"}:
            vx = require_number(payload, "vx")
            vy = require_number(payload, "vy")
            yaw_rate = require_number(payload, "yaw_rate")
            duration = require_number(payload, "duration_s")
            self._check_range("vx", vx, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
            self._check_range("vy", vy, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
            self._check_range("yaw_rate", yaw_rate, -MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS)
            self._check_range("duration_s", duration, 0.05, MAX_MOTION_DURATION_S)
            with self._lifecycle_lock:
                self._require_not_closing()
                self._require_motion_ready()
                return HTTPStatus.ACCEPTED, self._start_motion(
                    action,
                    vx=vx,
                    vy=vy,
                    yaw_rate=yaw_rate,
                    duration_s=duration,
                )

        if action == "update_move":
            action_id = payload.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                raise ApiError(HTTPStatus.BAD_REQUEST, "action_id must be a string")
            vx = require_number(payload, "vx")
            vy = require_number(payload, "vy")
            yaw_rate = require_number(payload, "yaw_rate")
            self._check_range("vx", vx, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
            self._check_range("vy", vy, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
            self._check_range("yaw_rate", yaw_rate, -MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS)
            with self._lifecycle_lock:
                self._require_not_closing()
                self._require_motion_ready()
                return HTTPStatus.OK, self._update_move(
                    action_id=action_id,
                    vx=vx,
                    vy=vy,
                    yaw_rate=yaw_rate,
                )

        if action == "move_distance":
            distance = require_number(payload, "distance_m")
            speed = abs(require_number(payload, "speed_mps"))
            if distance == 0.0:
                raise ApiError(HTTPStatus.BAD_REQUEST, "distance_m must be non-zero")
            self._check_range("distance_m", distance, -MAX_MOVE_DISTANCE_M, MAX_MOVE_DISTANCE_M)
            self._check_range("speed_mps", speed, 0.05, MAX_LINEAR_SPEED_MPS)
            timeout = min(MAX_MOTION_DURATION_S, abs(distance) / speed + 2.0)
            with self._lifecycle_lock:
                self._require_not_closing()
                self._require_motion_ready()
                return HTTPStatus.ACCEPTED, self._start_motion(
                    action,
                    vx=math.copysign(speed, distance),
                    vy=0.0,
                    yaw_rate=0.0,
                    distance_m=abs(distance),
                    duration_s=timeout,
                )

        if action == "turn":
            angle = require_number(payload, "angle_rad")
            yaw_rate = abs(require_number(payload, "yaw_rate_rps"))
            if angle == 0.0:
                raise ApiError(HTTPStatus.BAD_REQUEST, "angle_rad must be non-zero")
            self._check_range("angle_rad", angle, -MAX_TURN_ANGLE_RAD, MAX_TURN_ANGLE_RAD)
            self._check_range("yaw_rate_rps", yaw_rate, 0.1, MAX_YAW_RATE_RPS)
            timeout = min(MAX_MOTION_DURATION_S, abs(angle) / yaw_rate + 2.0)
            with self._lifecycle_lock:
                self._require_not_closing()
                self._require_motion_ready()
                return HTTPStatus.ACCEPTED, self._start_motion(
                    action,
                    vx=0.0,
                    vy=0.0,
                    yaw_rate=math.copysign(yaw_rate, angle),
                    angle_rad=abs(angle),
                    duration_s=timeout,
                )

        raise ApiError(HTTPStatus.BAD_REQUEST, f"unsupported action: {action}")

    @staticmethod
    def _check_range(name: str, value: float, minimum: float, maximum: float) -> None:
        if not minimum <= value <= maximum:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{name} must be between {minimum} and {maximum}",
            )

    def _require_not_closing(self) -> None:
        with self._lock:
            if self._closing or self._closed:
                raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "executor is closing")

    def _require_fresh_state(self) -> RobotState:
        state = self.transport.state()
        if state is None:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "robot state is unavailable")
        if (
            isinstance(state.sequence, bool)
            or not isinstance(state.sequence, int)
            or state.sequence < 1
            or not math.isfinite(state.received_at)
        ):
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "robot state has an invalid sequence or timestamp",
            )
        state_age_s = time.monotonic() - state.received_at
        if state_age_s < 0.0 or state_age_s > STATE_MAX_AGE_S:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "robot state is stale")
        return state

    def _require_motion_ready(self) -> RobotState:
        if not self.operator_motion_ready:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "operator motion-ready interlock is not enabled",
            )
        state = self._require_fresh_state()
        numeric_state = (
            *state.position,
            state.roll,
            state.pitch,
            state.yaw,
            *state.velocity,
            state.yaw_rate,
        )
        if (
            len(state.position) < 2
            or len(state.velocity) < 2
            or not all(math.isfinite(value) for value in numeric_state)
        ):
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "robot state contains invalid numeric data",
            )
        if state.error_code != 0:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"robot reports error code {state.error_code}",
            )
        if abs(state.roll) > MAX_TILT_RAD or abs(state.pitch) > MAX_TILT_RAD:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "robot tilt exceeds the motion safety limit",
            )
        return state

    def _execute_posture(self, action: str) -> dict[str, Any]:
        with self._lifecycle_lock:
            self._require_not_closing()
            if not self.operator_motion_ready:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "operator motion-ready interlock is not enabled",
                )
            stop_result = self._cancel_and_stop_locked("posture_change")
            if not stop_result["worker_joined"] or not stop_result["stopped"]:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "could not confirm cancellation before posture change",
                )
            self._require_fresh_state()
            # A posture RPC can have an uncertain outcome on timeout, so revoke
            # the operator attestation before sending it.
            with self._lock:
                self.operator_motion_ready = False
            with self._command_lock:
                code = self.transport.posture(action)
            if code != 0:
                raise ApiError(HTTPStatus.BAD_GATEWAY, f"SDK returned code {code}")
            return {
                "accepted": True,
                "action": action,
                "sdk_code": code,
                "operator_motion_ready": False,
            }

    def _start_motion(self, action: str, **parameters: float) -> dict[str, Any]:
        with self._lock:
            if self._worker and self._worker.is_alive():
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "another motion is active; call stop first",
                )
            action_id = uuid.uuid4().hex
            self._stop_event = threading.Event()
            self._active = {
                "id": action_id,
                "action": action,
                "parameters": parameters,
                "started_at_unix": time.time(),
            }
            if action == "stream_move":
                self._active["heartbeat_at_monotonic"] = time.monotonic()
            self._worker = threading.Thread(
                target=self._motion_worker,
                args=(action_id, action, parameters, self._stop_event),
                daemon=True,
                name=f"go2-{action}",
            )
            self._worker.start()
            return {"accepted": True, "action_id": action_id, "action": action}

    def _update_move(
        self,
        *,
        action_id: str,
        vx: float,
        vy: float,
        yaw_rate: float,
    ) -> dict[str, Any]:
        """Update velocity without a StopMove gap or extending the deadline."""

        with self._lock:
            if not self._active or not self._worker or not self._worker.is_alive():
                raise ApiError(HTTPStatus.CONFLICT, "no timed move is active")
            if self._active.get("id") != action_id:
                raise ApiError(HTTPStatus.CONFLICT, "action_id is not active")
            if self._active.get("action") not in {"move", "stream_move"}:
                raise ApiError(HTTPStatus.CONFLICT, "only a timed move can be updated")
            parameters = self._active.get("parameters")
            if not isinstance(parameters, dict):
                raise ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "active move parameters are invalid",
                )
            parameters.update(vx=vx, vy=vy, yaw_rate=yaw_rate)
            self._active["updated_at_unix"] = time.time()
            if self._active.get("action") == "stream_move":
                self._active["heartbeat_at_monotonic"] = time.monotonic()
            return {
                "accepted": True,
                "action_id": action_id,
                "action": "update_move",
                "vx": vx,
                "vy": vy,
                "yaw_rate": yaw_rate,
            }

    def _motion_worker(
        self,
        action_id: str,
        action: str,
        parameters: dict[str, float],
        stop_event: threading.Event,
    ) -> None:
        status = "completed"
        detail = "target reached"
        started = time.monotonic()
        start_state: RobotState | None = None
        previous_yaw = 0.0
        accumulated_yaw = 0.0
        travelled_distance = 0.0
        turn_direction = math.copysign(1.0, parameters["yaw_rate"])
        try:
            start_state = self._require_motion_ready()
            previous_yaw = start_state.yaw
            while not stop_event.is_set():
                elapsed = time.monotonic() - started
                if elapsed >= parameters["duration_s"]:
                    if action in {"move_distance", "turn"}:
                        status, detail = "failed", "safety timeout before target"
                    else:
                        detail = "duration elapsed"
                    break

                current = self._require_motion_ready()
                if action == "stream_move":
                    with self._lock:
                        active = self._active
                        heartbeat_at = (
                            active.get("heartbeat_at_monotonic")
                            if active and active.get("id") == action_id
                            else None
                        )
                    if (
                        isinstance(heartbeat_at, bool)
                        or not isinstance(heartbeat_at, (int, float))
                        or time.monotonic() - float(heartbeat_at) > STREAM_HEARTBEAT_TIMEOUT_S
                    ):
                        status, detail = "failed", "stream heartbeat expired"
                        break
                if action == "move_distance":
                    assert start_state is not None
                    travelled_distance = math.hypot(
                        current.position[0] - start_state.position[0],
                        current.position[1] - start_state.position[1],
                    )
                    if travelled_distance >= parameters["distance_m"]:
                        break
                elif action == "turn":
                    signed_delta = turn_direction * normalize_angle(current.yaw - previous_yaw)
                    accumulated_yaw = max(0.0, accumulated_yaw + signed_delta)
                    previous_yaw = current.yaw
                    if accumulated_yaw >= parameters["angle_rad"]:
                        break

                with self._lock:
                    vx = parameters["vx"]
                    vy = parameters["vy"]
                    yaw_rate = parameters["yaw_rate"]
                with self._command_lock:
                    # stop() sets the event before waiting for this lock. This
                    # closes the StopMove/Move race.
                    if stop_event.is_set():
                        status, detail = "cancelled", "stop requested"
                        break
                    code = self.transport.move(vx, vy, yaw_rate)
                if code != 0:
                    status, detail = "failed", f"SDK Move returned code {code}"
                    break
                stop_event.wait(CONTROL_PERIOD_S)
            if stop_event.is_set():
                status, detail = "cancelled", "stop requested"
        except Exception as exc:  # noqa: BLE001 - worker must always reach StopMove
            status, detail = "failed", str(exc)
        finally:
            try:
                with self._command_lock:
                    code = self.transport.stop()
                if code != 0 and status == "completed":
                    status, detail = "failed", f"SDK StopMove returned code {code}"
            except Exception as exc:  # noqa: BLE001 - report final safety-stop failure
                status, detail = "failed", f"stop failed: {exc}"
            result: dict[str, Any] = {
                "id": action_id,
                "action": action,
                "status": status,
                "detail": detail,
                "elapsed_s": round(time.monotonic() - started, 3),
                "finished_at_unix": time.time(),
            }
            if action == "move_distance":
                result["measured_distance_m"] = round(travelled_distance, 5)
            elif action == "turn":
                result["measured_angle_rad"] = round(accumulated_yaw, 5)
            with self._lock:
                self._last_result = result
                if self._active and self._active["id"] == action_id:
                    self._active = None

    def _cancel_and_stop_locked(self, reason: str) -> dict[str, Any]:
        with self._lock:
            active_id = self._active["id"] if self._active else None
            stop_event = self._stop_event
            worker = self._worker
            stop_event.set()

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=WORKER_STOP_JOIN_TIMEOUT_S)
        worker_joined = worker is None or not worker.is_alive()

        stop_error = None
        try:
            with self._command_lock:
                code = self.transport.stop()
        except Exception as exc:  # noqa: BLE001 - emergency stop returns structured status
            code = -1
            stop_error = str(exc)
        return {
            "stopped": code == 0 and worker_joined,
            "reason": reason,
            "cancelled_action_id": active_id,
            "sdk_code": code,
            "worker_joined": worker_joined,
            "error": stop_error,
        }

    def stop(self, reason: str = "stop") -> dict[str, Any]:
        with self._lifecycle_lock:
            result = self._cancel_and_stop_locked(reason)
            # Re-arming requires an intentional process restart with
            # --operator-ready; there is no remote re-arm endpoint.
            with self._lock:
                self.operator_motion_ready = False
            return {**result, "operator_motion_ready": False}

    def close(self, send_stop: bool = True) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    return
                self._closing = True
            try:
                self._cancel_and_stop_locked("close" if send_stop else "close_after_signal")
                with self._lock:
                    self.operator_motion_ready = False
            finally:
                try:
                    self.transport.close()
                finally:
                    with self._lock:
                        self._closed = True


__all__ = ["ActionExecutor"]
