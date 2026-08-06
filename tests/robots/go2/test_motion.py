from __future__ import annotations

from types import SimpleNamespace

from embodied_runtime.robots.go2 import BaseVelocityCommand, Go2VelocityLease


class _Client:
    def __init__(self) -> None:
        self.started: list[BaseVelocityCommand] = []
        self.updated: list[tuple[str, BaseVelocityCommand]] = []
        self.stop_calls = 0

    def start_velocity_lease(self, command: BaseVelocityCommand, *, duration_s: float) -> str:
        assert duration_s == 10.0
        self.started.append(command)
        return "lease-1"

    def update_velocity_lease(self, action_id: str, command: BaseVelocityCommand) -> None:
        self.updated.append((action_id, command))

    def state(self):
        return SimpleNamespace(active_action={"id": "lease-1"})

    def stop(self) -> None:
        self.stop_calls += 1


def test_velocity_lease_starts_once_then_heartbeats_without_command_gaps() -> None:
    client = _Client()
    lease = Go2VelocityLease(client)  # type: ignore[arg-type]
    moving = BaseVelocityCommand(0.4, 0.0, 0.0)

    lease.send(BaseVelocityCommand.stopped())
    lease.send(moving)
    lease.send(moving)
    lease.send(BaseVelocityCommand.stopped())
    lease.stop()

    assert client.started == [moving]
    assert [item[1] for item in client.updated] == [moving, BaseVelocityCommand.stopped()]
    assert client.stop_calls == 1
