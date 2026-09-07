"""Per-robot command arbitration for the control service runtime."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from rlinf_deploy.robots import RobotAction, RobotAdapter


class CommandSource(str, Enum):
    MODEL = "model"
    MANUAL = "manual"


class AuthorityState(str, Enum):
    MODEL = "model"
    MANUAL = "manual"
    ESTOP_LATCHED = "estop_latched"


class CommandStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    CLEARED = "cleared"
    REPLACED = "replaced"
    EXPIRED = "expired"
    FAILED = "failed"


_TERMINAL_STATUSES = frozenset(
    {
        CommandStatus.EXECUTED,
        CommandStatus.CANCELLED,
        CommandStatus.CLEARED,
        CommandStatus.REPLACED,
        CommandStatus.EXPIRED,
        CommandStatus.FAILED,
    }
)


class CommandRejected(RuntimeError):
    """The current authority state rejected a command."""


class CommandCancelled(RuntimeError):
    """A command did not execute because authority changed."""


class EmergencyStopPolicy(str, Enum):
    """How emergency stop shares a port with an in-flight execute call."""

    SERIALIZED = "serialized"
    PREEMPTIVE = "preemptive"


@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    source: CommandSource
    action: RobotAction
    deadline_s: float | None = None


class CommandTicket:
    """Observable completion state for one submitted command."""

    def __init__(self, envelope: CommandEnvelope) -> None:
        self.envelope = envelope
        self.cancel_event = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._status = CommandStatus.QUEUED
        self._error: BaseException | None = None

    @property
    def status(self) -> CommandStatus:
        with self._lock:
            return self._status

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def wait(self, timeout_s: float | None = None) -> CommandTicket:
        if not self._done.wait(timeout_s):
            raise TimeoutError("command did not finish before the timeout")
        return self

    def raise_for_failure(self) -> None:
        status = self.status
        if status is CommandStatus.FAILED:
            error = self.error
            if error is not None:
                raise error
            raise RuntimeError("command failed without an error")
        if status is not CommandStatus.EXECUTED:
            raise CommandCancelled(f"command ended with status {status.value}")

    def _mark_running(self) -> bool:
        with self._lock:
            if self._status is not CommandStatus.QUEUED:
                return False
            self._status = CommandStatus.RUNNING
            return True

    def _finish(
        self,
        status: CommandStatus,
        error: BaseException | None = None,
    ) -> None:
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"{status.value} is not a terminal command status")
        with self._lock:
            if self._status in _TERMINAL_STATUSES:
                return
            self._status = status
            self._error = error
            self._done.set()


class RobotCommandPort(Protocol):
    robot_id: str

    def execute(
        self,
        action: RobotAction,
        cancel_event: threading.Event,
    ) -> None: ...

    def hold(self) -> None: ...

    def emergency_stop(self) -> None: ...


class RobotAdapterCommandPort:
    """Adapt one synchronous RobotAdapter to the arbiter port.

    SERIALIZED keeps execute, hold, and emergency_stop mutually exclusive for
    buses that cannot interleave stop traffic with motion commands. PREEMPTIVE
    is an explicit composition choice for adapters whose stop operation may run
    during an in-flight execute call.
    """

    def __init__(
        self,
        robot: RobotAdapter,
        *,
        action_resolver: Callable[[RobotAdapter, RobotAction], RobotAction] | None = None,
        emergency_stop_policy: EmergencyStopPolicy = (EmergencyStopPolicy.SERIALIZED),
    ) -> None:
        self.robot = robot
        self.robot_id = robot.robot_id
        self._action_resolver = action_resolver
        self._emergency_stop_policy = emergency_stop_policy
        self._command_lock = threading.Lock()
        self._stop_lock = threading.Lock()

    def execute(
        self,
        action: RobotAction,
        cancel_event: threading.Event,
    ) -> None:
        with self._command_lock:
            if cancel_event.is_set():
                return
            if self._action_resolver is not None:
                action = self._action_resolver(self.robot, action)
                if cancel_event.is_set():
                    return
            self.robot.execute(action)

    def hold(self) -> None:
        with self._command_lock, self._stop_lock:
            self.robot.stop()

    def emergency_stop(self) -> None:
        if self._emergency_stop_policy is EmergencyStopPolicy.PREEMPTIVE:
            with self._stop_lock:
                self.robot.stop()
        else:
            with self._command_lock, self._stop_lock:
                self.robot.stop()


class ArbiterCommandSink:
    """Runtime-facing model command sink backed by RobotControlArbiter."""

    def __init__(
        self, arbiter: RobotControlArbiter, cancel_event: threading.Event | None = None
    ) -> None:
        self.arbiter = arbiter
        self.robot_id = arbiter.robot_id
        self.cancel_event = cancel_event

    def execute(self, action: RobotAction) -> None:
        ticket = self.arbiter.submit_model(action, wait=True, task_cancel=self.cancel_event)
        ticket.raise_for_failure()

    def stop(self) -> None:
        self.arbiter.cancel_model_work()


class RobotControlArbiter:
    """Own one robot's model/manual authority, cancellation, and estop latch."""

    def __init__(
        self,
        port: RobotCommandPort,
        *,
        model_queue_capacity: int = 1,
        manual_deadman_timeout_s: float = 0.25,
        clock=time.monotonic,
    ) -> None:
        if model_queue_capacity < 1:
            raise ValueError("model_queue_capacity must be at least 1")
        if manual_deadman_timeout_s <= 0:
            raise ValueError("manual_deadman_timeout_s must be positive")
        self.port = port
        self.robot_id = port.robot_id
        self._model_queue_capacity = model_queue_capacity
        self._manual_deadman_timeout_s = manual_deadman_timeout_s
        self._clock = clock
        self._condition = threading.Condition(threading.Lock())
        self._authority = AuthorityState.MODEL
        self._deadman_active = False
        self._manual_heartbeat_s: float | None = None
        self._model_queue: deque[CommandTicket] = deque()
        self._manual_pending: CommandTicket | None = None
        self._active: CommandTicket | None = None
        self._hold_in_progress = False
        self._estop_generation = 0
        self._last_stop_error: str | None = None
        self._last_command_error: str | None = None
        self._model_task_cancel: threading.Event | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"{self.robot_id}-command-worker",
        )
        self._watchdog = threading.Thread(
            target=self._watch_deadman,
            daemon=True,
            name=f"{self.robot_id}-deadman-watchdog",
        )
        self._worker.start()
        self._watchdog.start()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "robot_id": self.robot_id,
                "authority": self._authority.value,
                "deadman_active": self._deadman_active,
                "active_source": (self._active.envelope.source.value if self._active else None),
                "active_status": self._active.status.value if self._active else None,
                "pending_model": len(self._model_queue),
                "manual_pending": self._manual_pending is not None,
                "hold_in_progress": self._hold_in_progress,
                "last_stop_error": self._last_stop_error,
                "last_error": self._last_command_error or self._last_stop_error,
                "closed": self._closed,
            }

    def begin_model_task(self) -> threading.Event:
        """Register cancellation before inference; takeover invalidates late results."""
        with self._condition:
            self._require_open()
            if self._authority is not AuthorityState.MODEL:
                raise CommandRejected("model control is not available")
            if self._model_task_cancel is not None:
                self._model_task_cancel.set()
            self._model_task_cancel = threading.Event()
            return self._model_task_cancel

    def submit_model(
        self,
        action: RobotAction,
        *,
        wait: bool = True,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        task_cancel: threading.Event | None = None,
    ) -> CommandTicket:
        return self.submit(
            action,
            source=CommandSource.MODEL,
            wait=wait,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
            task_cancel=task_cancel,
        )

    def submit_manual(
        self,
        action: RobotAction,
        *,
        wait: bool = False,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
    ) -> CommandTicket:
        return self.submit(
            action,
            source=CommandSource.MANUAL,
            wait=wait,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
        )

    def submit(
        self,
        action: RobotAction,
        *,
        source: CommandSource,
        wait: bool = True,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        task_cancel: threading.Event | None = None,
    ) -> CommandTicket:
        envelope = CommandEnvelope(source=source, action=action, deadline_s=deadline_s)
        ticket = CommandTicket(envelope)
        with self._condition:
            self._require_open()
            if task_cancel is not None and task_cancel.is_set():
                raise CommandCancelled("model task was cancelled")
            if self._authority is AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is latched")
            if source is CommandSource.MANUAL:
                if self._authority is not AuthorityState.MANUAL:
                    raise CommandRejected("manual control is not acquired")
                if not self._deadman_active:
                    raise CommandRejected("manual deadman is not active")
                self._manual_heartbeat_s = self._clock()
                if self._manual_pending is not None:
                    self._manual_pending._finish(CommandStatus.REPLACED)
                self._manual_pending = ticket
            else:
                if self._authority is AuthorityState.MANUAL:
                    raise CommandRejected("manual control owns the robot")
                if len(self._model_queue) >= self._model_queue_capacity:
                    raise CommandRejected("model command queue is full")
                self._model_queue.append(ticket)
            self._condition.notify_all()
        if wait:
            ticket.wait(timeout_s)
        return ticket

    def acquire_manual(self) -> None:
        with self._condition:
            self._require_open()
            if self._authority is AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is latched")
            self._authority = AuthorityState.MANUAL
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._cancel_active()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._clear_manual(CommandStatus.CLEARED)
            self._hold_in_progress = True
            self._condition.notify_all()
        self._finish_hold()

    def release_manual(self) -> None:
        with self._condition:
            self._require_open()
            if self._authority is AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is latched")
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._cancel_active(CommandSource.MANUAL)
            self._clear_manual(CommandStatus.CLEARED)
            self._hold_in_progress = True
            self._condition.notify_all()
        self._finish_hold(next_authority=AuthorityState.MODEL)

    def set_deadman(self, active: bool) -> None:
        should_hold = False
        with self._condition:
            self._require_open()
            if self._authority is not AuthorityState.MANUAL:
                raise CommandRejected("manual control is not acquired")
            self._deadman_active = bool(active)
            self._manual_heartbeat_s = self._clock() if active else None
            if not active:
                self._cancel_active(CommandSource.MANUAL)
                self._clear_manual(CommandStatus.CLEARED)
                self._hold_in_progress = True
                should_hold = True
            self._condition.notify_all()
        if should_hold:
            self._finish_hold()

    def hold(self) -> None:
        with self._condition:
            self._require_open()
            self._cancel_active()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._clear_manual(CommandStatus.CLEARED)
            self._hold_in_progress = True
            self._condition.notify_all()
        self._finish_hold()

    def cancel_model_work(self) -> None:
        with self._condition:
            self._require_open()
            if self._model_task_cancel is not None:
                self._model_task_cancel.set()
            active = self._active
            active_is_model = active is not None and active.envelope.source is CommandSource.MODEL
            self._clear_model_queue(CommandStatus.CLEARED)
            if active_is_model:
                active.cancel_event.set()
            should_hold = self._authority is AuthorityState.MODEL
            if should_hold:
                self._hold_in_progress = True
            self._condition.notify_all()
        if should_hold:
            self._finish_hold()

    def emergency_stop(self) -> None:
        with self._condition:
            self._require_open()
            self._estop_generation += 1
            self._authority = AuthorityState.ESTOP_LATCHED
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._cancel_active()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._clear_manual(CommandStatus.CLEARED)
            self._condition.notify_all()
        try:
            self.port.emergency_stop()
        except Exception as error:
            with self._condition:
                self._last_stop_error = str(error)
            raise

    def reset_emergency_stop(self) -> None:
        with self._condition:
            self._require_open()
            if self._authority is not AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is not latched")
            estop_generation = self._estop_generation
        try:
            self.port.hold()
        except Exception as error:
            with self._condition:
                self._last_stop_error = str(error)
            raise
        with self._condition:
            self._require_open()
            if self._estop_generation != estop_generation:
                raise CommandRejected("emergency stop was triggered during reset")
            self._authority = AuthorityState.MANUAL
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._last_stop_error = None
            self._last_command_error = None
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._cancel_active()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._clear_manual(CommandStatus.CLEARED)
            self._condition.notify_all()
        try:
            self.port.hold()
        finally:
            self._worker.join(timeout=1.0)
            self._watchdog.join(timeout=1.0)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("robot control arbiter is closed")

    def _finish_hold(
        self,
        *,
        next_authority: AuthorityState | None = None,
    ) -> None:
        try:
            self.port.hold()
        except Exception as error:
            with self._condition:
                self._estop_generation += 1
                self._authority = AuthorityState.ESTOP_LATCHED
                self._deadman_active = False
                self._manual_heartbeat_s = None
                self._hold_in_progress = False
                self._last_stop_error = str(error)
                self._cancel_active()
                self._clear_model_queue(CommandStatus.CLEARED)
                self._clear_manual(CommandStatus.CLEARED)
                self._condition.notify_all()
            raise
        with self._condition:
            if next_authority is not None and self._authority is not AuthorityState.ESTOP_LATCHED:
                self._authority = next_authority
            self._hold_in_progress = False
            if self._authority is not AuthorityState.ESTOP_LATCHED:
                self._last_stop_error = None
            self._condition.notify_all()

    def _cancel_active(self, source: CommandSource | None = None) -> None:
        if source in (None, CommandSource.MODEL) and self._model_task_cancel is not None:
            self._model_task_cancel.set()
        if self._active is None:
            return
        if source is None or self._active.envelope.source is source:
            self._active.cancel_event.set()

    def _clear_model_queue(self, status: CommandStatus) -> bool:
        cleared_any = False
        while self._model_queue:
            self._model_queue.popleft()._finish(status)
            cleared_any = True
        return cleared_any

    def _clear_manual(self, status: CommandStatus) -> None:
        if self._manual_pending is not None:
            self._manual_pending._finish(status)
            self._manual_pending = None

    def _next_ticket(self) -> CommandTicket | None:
        if self._hold_in_progress or self._authority is AuthorityState.ESTOP_LATCHED:
            return None
        if self._authority is AuthorityState.MANUAL:
            if not self._deadman_active:
                return None
            ticket = self._manual_pending
            self._manual_pending = None
            return ticket
        if self._model_queue:
            return self._model_queue.popleft()
        return None

    def _run(self) -> None:
        while True:
            with self._condition:
                ticket = self._next_ticket()
                while ticket is None and not self._closed:
                    self._condition.wait()
                    ticket = self._next_ticket()
                if self._closed:
                    return
                if not ticket._mark_running():
                    continue
                self._active = ticket
                deadline_s = ticket.envelope.deadline_s
                if deadline_s is not None and self._clock() > deadline_s:
                    self._active = None
                    ticket._finish(CommandStatus.EXPIRED)
                    continue
            try:
                self.port.execute(ticket.envelope.action, ticket.cancel_event)
            except Exception as error:
                status = (
                    CommandStatus.CANCELLED
                    if ticket.cancel_event.is_set()
                    else CommandStatus.FAILED
                )
                ticket._finish(status, error)
                with self._condition:
                    self._last_command_error = str(error)
                try:
                    self.emergency_stop()
                except Exception:
                    pass  # emergency_stop retains the stop error and latch.
            else:
                status = (
                    CommandStatus.CANCELLED
                    if ticket.cancel_event.is_set()
                    else CommandStatus.EXECUTED
                )
                ticket._finish(status)
            finally:
                with self._condition:
                    if self._active is ticket:
                        self._active = None
                    self._condition.notify_all()

    def _watch_deadman(self) -> None:
        while True:
            should_hold = False
            with self._condition:
                if self._closed:
                    return
                if (
                    self._authority is not AuthorityState.MANUAL
                    or not self._deadman_active
                    or self._manual_heartbeat_s is None
                ):
                    self._condition.wait()
                    continue
                remaining = (
                    self._manual_heartbeat_s + self._manual_deadman_timeout_s - self._clock()
                )
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                self._deadman_active = False
                self._manual_heartbeat_s = None
                self._cancel_active(CommandSource.MANUAL)
                self._clear_manual(CommandStatus.CLEARED)
                self._hold_in_progress = True
                self._condition.notify_all()
                should_hold = True
            if should_hold:
                try:
                    self._finish_hold()
                except Exception:
                    continue


__all__ = [
    "ArbiterCommandSink",
    "AuthorityState",
    "CommandCancelled",
    "CommandEnvelope",
    "CommandRejected",
    "CommandSource",
    "CommandStatus",
    "CommandTicket",
    "EmergencyStopPolicy",
    "RobotAdapterCommandPort",
    "RobotCommandPort",
    "RobotControlArbiter",
]
