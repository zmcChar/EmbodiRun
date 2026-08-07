from __future__ import annotations

import json

from embodied_runtime.deployment.unitree.go2.cli import main
from embodied_runtime.deployment.unitree.go2.transport import CommandResult, SshConnection

from .fakes import FakeTransport


def test_cli_reads_ssh_password_from_environment_without_exposing_it(capsys) -> None:
    password = "top-secret-password"
    captured: list[SshConnection] = []
    transport = FakeTransport([CommandResult(0, json.dumps({"python": "/usr/bin/python3"}))])

    def factory(connection: SshConnection) -> FakeTransport:
        captured.append(connection)
        return transport

    exit_code = main(
        ["--host", "go2.local", "--username", "unitree", "probe"],
        environ={"GO2_SSH_PASSWORD": password},
        transport_factory=factory,
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert captured[0].password == password
    assert password not in repr(captured[0])
    assert password not in output.out
    assert password not in output.err
    assert transport.closed


def test_cli_supports_interactive_password(capsys) -> None:
    password = "interactive-secret"
    captured: list[SshConnection] = []
    transport = FakeTransport([CommandResult(0, json.dumps({"python": "/usr/bin/python3"}))])

    def factory(connection: SshConnection) -> FakeTransport:
        captured.append(connection)
        return transport

    assert (
        main(
            [
                "--host",
                "go2.local",
                "--username",
                "unitree",
                "--ask-password",
                "probe",
            ],
            environ={},
            password_prompt=lambda _prompt: password,
            transport_factory=factory,
        )
        == 0
    )
    assert captured[0].password == password
    assert password not in capsys.readouterr().out


def test_cli_supports_key_auth_without_password() -> None:
    captured: list[SshConnection] = []
    transport = FakeTransport([CommandResult(0, json.dumps({"python": "/usr/bin/python3"}))])

    def factory(connection: SshConnection) -> FakeTransport:
        captured.append(connection)
        return transport

    assert (
        main(
            [
                "--host",
                "go2.local",
                "--username",
                "unitree",
                "--identity-file",
                "/tmp/go2_key",
                "probe",
            ],
            environ={},
            transport_factory=factory,
        )
        == 0
    )
    assert captured[0].password is None
    assert str(captured[0].identity_file) == "/tmp/go2_key"


def test_cli_redacts_password_from_errors(capsys) -> None:
    password = "literal-super-secret"

    class FailingTransport(FakeTransport):
        def resolve_path(self, path: str) -> str:
            raise RuntimeError(f"transport accidentally echoed {password}")

    assert (
        main(
            [
                "--host",
                "go2.local",
                "--username",
                "unitree",
                "--password",
                password,
                "probe",
            ],
            environ={},
            transport_factory=lambda _connection: FailingTransport(),
        )
        == 1
    )
    error = capsys.readouterr().err
    assert password not in error
    assert "<redacted>" in error


def test_start_tokens_are_loaded_from_environment_and_not_printed(capsys) -> None:
    api_token = "a" * 32
    transport = FakeTransport([CommandResult(0, "87\n")])

    assert (
        main(
            [
                "--host",
                "go2.local",
                "--username",
                "unitree",
                "start",
                "--service",
                "control",
                "--cyclonedds-lib-dir",
                "/opt/dds",
                "--control-bind",
                "0.0.0.0",
            ],
            environ={"GO2_API_TOKEN": api_token},
            transport_factory=lambda _connection: transport,
        )
        == 0
    )
    output = capsys.readouterr()
    assert api_token not in output.out
    assert api_token not in output.err
    assert (
        api_token.encode()
        in transport.files["/home/tester/.local/share/rlinf-go2-agent/run/control.env"]
    )


def test_start_uses_ssh_host_as_default_service_bind() -> None:
    api_token = "a" * 32
    transport = FakeTransport([CommandResult(0, "88\n")])

    assert (
        main(
            [
                "--host",
                "192.0.2.34",
                "--username",
                "unitree",
                "start",
                "--service",
                "control",
                "--cyclonedds-lib-dir",
                "/opt/dds",
            ],
            environ={"GO2_API_TOKEN": api_token},
            transport_factory=lambda _connection: transport,
        )
        == 0
    )

    start_command = transport.commands[0][0]
    assert "--host 192.0.2.34" in start_command
