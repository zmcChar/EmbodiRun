"""Heartbeat-managed velocity lease for the bounded Go2 control API."""

from __future__ import annotations

from .client import Go2ClientError, Go2ControlClient
from .types import BaseVelocityCommand


class Go2VelocityLease:
    def __init__(self, client: Go2ControlClient, *, duration_s: float = 10.0) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.client = client
        self.duration_s = float(duration_s)
        self.action_id: str | None = None
        self.last_command = BaseVelocityCommand.stopped()

    def _start(self, command: BaseVelocityCommand) -> None:
        self.action_id = self.client.start_velocity_lease(
            command,
            duration_s=self.duration_s,
        )

    def send(self, command: BaseVelocityCommand) -> None:
        if not isinstance(command, BaseVelocityCommand):
            raise TypeError("command must be a BaseVelocityCommand")
        if self.action_id is None:
            if command.moving:
                self._start(command)
            self.last_command = command
            return
        try:
            self.client.update_velocity_lease(self.action_id, command)
        except Go2ClientError:
            state = self.client.state()
            active_id = None if state.active_action is None else state.active_action.get("id")
            if active_id == self.action_id:
                raise
            self.action_id = None
            if command.moving:
                self._start(command)
        self.last_command = command

    def stop(self) -> None:
        self.client.stop()
        self.action_id = None
        self.last_command = BaseVelocityCommand.stopped(self.last_command.limits)


__all__ = ["Go2VelocityLease"]
