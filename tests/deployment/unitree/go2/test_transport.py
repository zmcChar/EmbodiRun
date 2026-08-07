from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_runtime.deployment.unitree.go2.transport import ParamikoTransport, SshConnection


class _Channel:
    def recv_exit_status(self) -> int:
        return 0


class _Stream:
    channel = _Channel()

    def read(self) -> bytes:
        return b""


class _Client:
    def __init__(self, password_to_echo: str | None = None) -> None:
        self.password_to_echo = password_to_echo
        self.loaded_system_keys = False
        self.policies: list[object] = []
        self.connect_kwargs: dict[str, object] = {}
        self.closed = False

    def load_system_host_keys(self) -> None:
        self.loaded_system_keys = True

    def set_missing_host_key_policy(self, policy: object) -> None:
        self.policies.append(policy)

    def connect(self, **kwargs: object) -> None:
        self.connect_kwargs = kwargs
        if self.password_to_echo is not None:
            raise RuntimeError(f"bad password: {self.password_to_echo}")

    def exec_command(self, _command: str, *, timeout: float):
        assert timeout > 0
        return None, _Stream(), _Stream()

    def close(self) -> None:
        self.closed = True


class _AutoAddPolicy:
    pass


def _paramiko_module(client: _Client) -> SimpleNamespace:
    return SimpleNamespace(SSHClient=lambda: client, AutoAddPolicy=_AutoAddPolicy)


def test_unknown_host_keys_are_rejected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    monkeypatch.setitem(sys.modules, "paramiko", _paramiko_module(client))
    transport = ParamikoTransport(SshConnection(host="go2.local", username="unitree"))

    transport.run("true")

    assert client.loaded_system_keys
    assert client.policies == []


def test_accept_new_host_key_requires_explicit_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    monkeypatch.setitem(sys.modules, "paramiko", _paramiko_module(client))
    transport = ParamikoTransport(
        SshConnection(
            host="go2.local",
            username="unitree",
            accept_new_host_key=True,
        )
    )

    transport.run("true")

    assert len(client.policies) == 1
    assert isinstance(client.policies[0], _AutoAddPolicy)


def test_paramiko_connection_errors_redact_password() -> None:
    password = "do-not-echo-this"
    client = _Client(password_to_echo=password)
    transport = ParamikoTransport(
        SshConnection(host="go2.local", username="unitree", password=password),
        client_factory=lambda: client,
    )

    with pytest.raises(RuntimeError) as caught:
        transport.run("true")

    assert password not in str(caught.value)
    assert password not in repr(caught.value)
    assert "<redacted>" in str(caught.value)


def test_identity_file_disables_implicit_key_discovery() -> None:
    client = _Client()
    connection = SshConnection(
        host="go2.local",
        username="unitree",
        identity_file=Path("/tmp/go2-key"),
    )
    transport = ParamikoTransport(connection, client_factory=lambda: client)

    transport.run("true")

    assert client.connect_kwargs["key_filename"] == "/tmp/go2-key"
    assert client.connect_kwargs["look_for_keys"] is False
