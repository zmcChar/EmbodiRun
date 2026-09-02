"""SSH deployment command execution."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from ..config import ConnectionConfig
from .command import Command, CommandError, CommandResult


class SshExecutor:
    """Execute POSIX commands over one lazily-created Paramiko connection."""

    def __init__(self, connection: ConnectionConfig) -> None:
        if connection.kind != "ssh":
            raise ValueError("SshExecutor requires an SSH connection")
        self.connection = connection
        self._client: Any | None = None
        self._password = _password_from_environment(connection)

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import paramiko
        except ImportError as error:  # pragma: no cover - depends on installed group
            raise RuntimeError(
                "SSH execution requires the host environment; run "
                "`uv sync --frozen --no-dev --group host`"
            ) from error

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        if self.connection.accept_new_host_key:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict[str, Any] = {
            "hostname": self.connection.host,
            "port": self.connection.port,
            "username": self.connection.username,
            "timeout": self.connection.connect_timeout_s,
            "banner_timeout": self.connection.connect_timeout_s,
            "auth_timeout": self.connection.connect_timeout_s,
            "allow_agent": True,
            "look_for_keys": self.connection.identity_file is None,
        }
        if self._password is not None:
            kwargs["password"] = self._password
        if self.connection.identity_file is not None:
            kwargs["key_filename"] = os.fspath(
                Path(self.connection.identity_file).expanduser()
            )
        try:
            client.connect(**kwargs)
        except Exception as error:  # noqa: BLE001 - normalize third-party errors
            detail = _redact(str(error), self._password)
            raise RuntimeError(f"SSH connection failed: {detail}") from None
        self._client = client
        return client

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        script = render_posix(command)
        timeout = command.timeout_s or self.connection.command_timeout_s
        try:
            _stdin, stdout, stderr = self._connect().exec_command(
                script,
                timeout=timeout,
            )
            output = stdout.read().decode("utf-8", errors="replace")
            error = stderr.read().decode("utf-8", errors="replace")
            exit_code = int(stdout.channel.recv_exit_status())
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            detail = _redact(str(exc), self._password)
            raise RuntimeError(f"SSH command failed: {detail}") from None
        result = CommandResult(
            exit_code=exit_code,
            stdout=_redact(output, self._password),
            stderr=_redact(error, self._password),
        )
        if check and result.exit_code != 0:
            raise CommandError(result.exit_code, result.stderr)
        return result

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def render_posix(command: Command) -> str:
    """Render an argv-based command for a POSIX remote shell."""

    invocation = shlex.join(command.argv)
    if command.environment:
        assignments = " ".join(
            f"{name}={shlex.quote(value)}"
            for name, value in sorted(command.environment.items())
        )
        invocation = f"env {assignments} {invocation}"
    if command.cwd is not None:
        invocation = f"cd {shlex.quote(command.cwd)} && {invocation}"
    return invocation


def _password_from_environment(connection: ConnectionConfig) -> str | None:
    if connection.password_env is None:
        return None
    password = os.environ.get(connection.password_env)
    if password is None:
        raise RuntimeError(
            f"SSH password environment variable {connection.password_env!r} is not set"
        )
    return password


def _redact(value: str, secret: str | None) -> str:
    if secret:
        return value.replace(secret, "<redacted>")
    return value


__all__ = ["SshExecutor", "render_posix"]
