"""Supervise long-running deployment processes on local or remote nodes."""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass
from typing import Literal

from ..executor import Command, Executor
from .errors import ServiceError
from .spec import ServiceSpec

_SERVICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class ProcessStatus:
    """Observed state of one supervised service process."""

    state: Literal["stopped", "running", "stale", "pid-reused"]
    pid: int | None = None
    log: str | None = None


class ServiceSupervisor:
    """Manage long-running services through a local or SSH executor."""

    def __init__(self, executor: Executor, *, run_root: str, log_root: str) -> None:
        if not run_root or run_root == "/" or "\x00" in run_root:
            raise ValueError("run_root must be a non-root POSIX path")
        if not log_root or log_root == "/" or "\x00" in log_root:
            raise ValueError("log_root must be a non-root POSIX path")
        self.executor = executor
        self.run_root = run_root
        self.log_root = log_root

    def start(self, service: ServiceSpec) -> ProcessStatus:
        """Start a service idempotently and verify its process identity."""

        pid_file, log_file = self._paths(service.service_id)
        managed_command = _with_service_identity(service.command, service.service_id)
        script = "\n".join(
            [
                "set -eu",
                f"pid_file={shlex.quote(pid_file)}",
                f"log_file={shlex.quote(log_file)}",
                f"identity={shlex.quote(_identity(service.service_id))}",
                f"mkdir -p {shlex.quote(self.run_root)} {shlex.quote(self.log_root)}",
                'if [ -f "$pid_file" ]; then',
                '  old_pid=$(cat "$pid_file" 2>/dev/null || true)',
                "  case \"$old_pid\" in ''|*[!0-9]*) old_pid='' ;; esac",
                '  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then',
                '    if _rlinf_identity "$old_pid" "$identity"; then',
                "      printf 'running\\t%s\\n' \"$old_pid\"; exit 0",
                "    fi",
                "    printf '%s\\n' 'PID belongs to another process' >&2; exit 75",
                "  fi",
                '  rm -f "$pid_file"',
                "fi",
                "umask 077",
                f'nohup {_render_daemon(managed_command)} >>"$log_file" 2>&1 </dev/null &',
                "pid=$!",
                'printf \'%s\\n\' "$pid" >"$pid_file"',
                "sleep 0.2",
                'if ! kill -0 "$pid" 2>/dev/null; then',
                '  rm -f "$pid_file"',
                '  tail -n 20 "$log_file" >&2 || true',
                "  exit 1",
                "fi",
                'if ! _rlinf_identity "$pid" "$identity"; then',
                '  kill -TERM "$pid" 2>/dev/null || true',
                '  rm -f "$pid_file"',
                "  printf '%s\\n' 'started process identity could not be verified' >&2",
                "  exit 1",
                "fi",
                "printf 'running\\t%s\\n' \"$pid\"",
            ]
        )
        result = self.executor.run(_shell_command(_identity_function(script)))
        return _parse_status(result.stdout, log=log_file)

    def status(self, service_id: str) -> ProcessStatus:
        """Inspect the PID file without changing the remote process."""

        pid_file, log_file = self._paths(service_id)
        script = "\n".join(
            [
                "set -u",
                f"pid_file={shlex.quote(pid_file)}",
                f"identity={shlex.quote(_identity(service_id))}",
                "if [ ! -f \"$pid_file\" ]; then printf 'stopped\\n'; exit 0; fi",
                'pid=$(cat "$pid_file" 2>/dev/null || true)',
                "case \"$pid\" in ''|*[!0-9]*) printf 'stale\\n'; exit 0 ;; esac",
                'if ! kill -0 "$pid" 2>/dev/null; then',
                "  printf 'stale\\t%s\\n' \"$pid\"; exit 0",
                "fi",
                'if _rlinf_identity "$pid" "$identity"; then',
                "  printf 'running\\t%s\\n' \"$pid\"",
                "else",
                "  printf 'pid-reused\\t%s\\n' \"$pid\"",
                "fi",
            ]
        )
        result = self.executor.run(_shell_command(_identity_function(script)))
        return _parse_status(result.stdout, log=log_file)

    def stop(self, service_id: str) -> ProcessStatus:
        """Stop only a process carrying this service's identity marker."""

        pid_file, log_file = self._paths(service_id)
        script = "\n".join(
            [
                "set -u",
                f"pid_file={shlex.quote(pid_file)}",
                f"identity={shlex.quote(_identity(service_id))}",
                "if [ ! -f \"$pid_file\" ]; then printf 'stopped\\n'; exit 0; fi",
                'pid=$(cat "$pid_file" 2>/dev/null || true)',
                "case \"$pid\" in ''|*[!0-9]*)",
                "  rm -f \"$pid_file\"; printf 'stale\\n'; exit 0 ;; esac",
                'if ! kill -0 "$pid" 2>/dev/null; then',
                '  rm -f "$pid_file"; printf \'stale\\t%s\\n\' "$pid"; exit 0',
                "fi",
                'if ! _rlinf_identity "$pid" "$identity"; then',
                "  printf '%s\\n' 'PID belongs to another process; refusing to signal it' >&2",
                "  exit 75",
                "fi",
                'kill -TERM "$pid"',
                "count=0",
                'while kill -0 "$pid" 2>/dev/null && [ "$count" -lt 50 ]; do',
                "  sleep 0.1; count=$((count + 1))",
                "done",
                'if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid"; fi',
                'rm -f "$pid_file"',
                "printf 'stopped\\n'",
            ]
        )
        result = self.executor.run(
            _shell_command(_identity_function(script), timeout_s=10.0)
        )
        return _parse_status(result.stdout, log=log_file)

    def _paths(self, service_id: str) -> tuple[str, str]:
        _validate_service_id(service_id)
        return (
            posixpath.join(self.run_root, f"{service_id}.pid"),
            posixpath.join(self.log_root, f"{service_id}.log"),
        )


def _with_service_identity(command: Command, service_id: str) -> Command:
    environment = dict(command.environment)
    name = "RLINF_DEPLOY_SERVICE_ID"
    existing = environment.get(name)
    if existing is not None and existing != service_id:
        raise ServiceError(f"service command overrides reserved variable {name}")
    environment[name] = service_id
    return Command(
        argv=command.argv,
        cwd=command.cwd,
        environment=environment,
        timeout_s=command.timeout_s,
    )


def _render_daemon(command: Command) -> str:
    body = f"exec {shlex.join(command.argv)}"
    if command.cwd is not None:
        body = f"cd {shlex.quote(command.cwd)} && {body}"
    launcher = ["env"]
    launcher.extend(
        f"{name}={value}" for name, value in sorted(command.environment.items())
    )
    launcher.extend(("sh", "-c", body))
    return shlex.join(launcher)


def _identity(service_id: str) -> str:
    _validate_service_id(service_id)
    return f"RLINF_DEPLOY_SERVICE_ID={service_id}"


def _identity_function(script: str) -> str:
    function = (
        "_rlinf_identity() {\n"
        '  [ -r "/proc/$1/environ" ] &&\n'
        "    tr '\\000' '\\n' <\"/proc/$1/environ\" | grep -Fqx \"$2\"\n"
        "}"
    )
    return f"{function}\n{script}"


def _shell_command(script: str, *, timeout_s: float | None = None) -> Command:
    return Command(("sh", "-c", script), timeout_s=timeout_s)


def _parse_status(output: str, *, log: str) -> ProcessStatus:
    lines = [line for line in output.splitlines() if line]
    if not lines:
        raise RuntimeError("service supervisor returned no status")
    fields = lines[-1].split("\t")
    state = fields[0]
    if state not in {"stopped", "running", "stale", "pid-reused"}:
        raise RuntimeError(f"service supervisor returned invalid status {state!r}")
    pid: int | None = None
    if len(fields) > 1:
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise RuntimeError("service supervisor returned an invalid PID") from error
        if pid <= 0:
            raise RuntimeError("service supervisor returned an invalid PID")
    return ProcessStatus(state=state, pid=pid, log=log)


def _validate_service_id(service_id: str) -> None:
    if not _SERVICE_ID.fullmatch(service_id):
        raise ServiceError(
            "service IDs must start with an alphanumeric character and contain "
            "only alphanumerics, dots, underscores, or hyphens"
        )


__all__ = ["ProcessStatus", "ServiceSupervisor"]
