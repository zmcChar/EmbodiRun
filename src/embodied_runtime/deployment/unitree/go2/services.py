"""PID-backed lifecycle supervision for Unitree Go2 edge services."""

from __future__ import annotations

import posixpath
import shlex
from collections.abc import Sequence

from .options import (
    CameraStartOptions,
    ControlStartOptions,
    ServiceName,
    ServiceSelection,
    StartOptions,
    selected_services,
)
from .transport import RemoteTransport

SERVICE_MODULES: dict[ServiceName, str] = {
    "control": "embodied_runtime.robots.unitree.go2.agent.control",
    "camera": "embodied_runtime.robots.unitree.go2.agent.camera",
}


class Go2ServiceSupervisor:
    """Start, inspect, and stop the two robot-resident HTTP services."""

    def __init__(
        self,
        transport: RemoteTransport,
        *,
        source_root: str,
        run_root: str,
        log_root: str,
    ) -> None:
        self.transport = transport
        self.source_root = source_root
        self.run_root = run_root
        self.log_root = log_root

    def start(self, options: StartOptions) -> dict[str, object]:
        results: dict[str, object] = {}
        for service in selected_services(options.services):
            if service == "control":
                assert options.control is not None
                command, environment = self._control_command(options.python, options.control)
            else:
                assert options.camera is not None
                command, environment = self._camera_command(options.python, options.camera)
            results[service] = self._start_service(service, command, environment)
        return {"services": results}

    def status(self, services: ServiceSelection = "all") -> dict[str, object]:
        return {
            "services": {
                service: self._status_service(service) for service in selected_services(services)
            }
        }

    def stop(self, services: ServiceSelection = "all") -> dict[str, object]:
        selected = selected_services(services)
        selected.sort(key=lambda name: 0 if name == "control" else 1)
        return {"services": {service: self._stop_service(service) for service in selected}}

    def _control_command(
        self,
        python: str,
        options: ControlStartOptions,
    ) -> tuple[list[str], dict[str, str]]:
        command = [
            python,
            "-m",
            SERVICE_MODULES["control"],
            "--mode",
            options.mode,
            "--interface",
            options.interface,
            "--state-topic",
            options.state_topic,
            "--host",
            options.bind,
            "--port",
            str(options.port),
        ]
        if options.cyclonedds_lib_dir:
            command.extend(["--cyclonedds-lib-dir", options.cyclonedds_lib_dir])
        if options.operator_ready:
            command.append("--operator-ready")
        environment = {"PYTHONPATH": self.source_root}
        if options.api_token is not None:
            environment["GO2_API_TOKEN"] = options.api_token
        return command, environment

    def _camera_command(
        self,
        python: str,
        options: CameraStartOptions,
    ) -> tuple[list[str], dict[str, str]]:
        command = [
            python,
            "-m",
            SERVICE_MODULES["camera"],
            "--backend",
            options.backend,
            "--width",
            str(options.width),
            "--height",
            str(options.height),
            "--jpeg-fps",
            str(options.jpeg_fps),
            "--bind",
            options.bind,
            "--port",
            str(options.port),
        ]
        if options.backend == "realsense":
            assert options.depth_scale is not None
            command.extend(
                [
                    "--realsense-fps",
                    str(options.camera_fps),
                    "--depth-calibrated",
                    "--rgb-depth-aligned",
                    "--depth-scale",
                    str(options.depth_scale),
                ]
            )
            if options.realsense_serial:
                command.extend(["--realsense-serial", options.realsense_serial])
        else:
            command.extend(["--device", options.device])
        environment = {"PYTHONPATH": self.source_root}
        if options.camera_token is not None:
            environment["GO2_CAMERA_TOKEN"] = options.camera_token
        return command, environment

    def _start_service(
        self,
        service: ServiceName,
        command: Sequence[str],
        environment: dict[str, str],
    ) -> dict[str, object]:
        self.transport.make_dirs(self.run_root, mode=0o700)
        self.transport.make_dirs(self.log_root, mode=0o700)
        env_path = posixpath.join(self.run_root, f"{service}.env")
        env_payload = "".join(
            f"{name}={shlex.quote(value)}\n" for name, value in sorted(environment.items())
        )
        self.transport.put_bytes(env_payload.encode("utf-8"), env_path, mode=0o600)

        pid_path = posixpath.join(self.run_root, f"{service}.pid")
        log_path = posixpath.join(self.log_root, f"{service}.log")
        module = SERVICE_MODULES[service]
        script = "\n".join(
            [
                "set -eu",
                f"env_file={shlex.quote(env_path)}",
                f"pid_file={shlex.quote(pid_path)}",
                f"log_file={shlex.quote(log_path)}",
                "trap 'rm -f \"$env_file\"' EXIT HUP INT TERM",
                'if [ -f "$pid_file" ]; then',
                '  old_pid=$(cat "$pid_file" 2>/dev/null || true)',
                "  case \"$old_pid\" in ''|*[!0-9]*) old_pid='' ;; esac",
                '  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then',
                f"    printf '%s\\n' {shlex.quote(service + ' is already running')} >&2",
                "    exit 73",
                "  fi",
                '  rm -f "$pid_file"',
                "fi",
                "umask 077",
                "set -a",
                '. "$env_file"',
                "set +a",
                f'nohup {shlex.join(list(command))} >>"$log_file" 2>&1 </dev/null &',
                "pid=$!",
                'printf \'%s\\n\' "$pid" >"$pid_file"',
                "sleep 0.5",
                'if ! kill -0 "$pid" 2>/dev/null; then',
                '  rm -f "$pid_file"',
                '  tail -n 20 "$log_file" >&2 || true',
                "  exit 1",
                "fi",
                "cmdline=$(tr '\\000' ' ' </proc/\"$pid\"/cmdline 2>/dev/null || true)",
                f'case "$cmdline" in *{shlex.quote(module)}*) ;; *)',
                "  printf '%s\\n' 'started process identity could not be verified' >&2",
                '  kill -TERM "$pid" 2>/dev/null || true',
                '  rm -f "$pid_file"',
                "  exit 1",
                "esac",
                "printf '%s\\n' \"$pid\"",
            ]
        )
        result = self.transport.run(script)
        try:
            pid = int(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"{service} start returned an invalid PID") from exc
        return {"state": "running", "pid": pid, "log": log_path}

    def _status_service(self, service: ServiceName) -> dict[str, object]:
        pid_path = posixpath.join(self.run_root, f"{service}.pid")
        module = SERVICE_MODULES[service]
        script = "\n".join(
            [
                "set -u",
                f"pid_file={shlex.quote(pid_path)}",
                "if [ ! -f \"$pid_file\" ]; then printf 'stopped\\n'; exit 0; fi",
                'pid=$(cat "$pid_file" 2>/dev/null || true)',
                "case \"$pid\" in ''|*[!0-9]*) printf 'stale\\n'; exit 0 ;; esac",
                'if ! kill -0 "$pid" 2>/dev/null; then printf \'stale\\t%s\\n\' "$pid"; exit 0; fi',
                "cmdline=$(tr '\\000' ' ' </proc/\"$pid\"/cmdline 2>/dev/null || true)",
                f'case "$cmdline" in *{shlex.quote(module)}*)',
                "  printf 'running\\t%s\\n' \"$pid\" ;;",
                "  *) printf 'pid-reused\\t%s\\n' \"$pid\" ;;",
                "esac",
            ]
        )
        output = self.transport.run(script).stdout.strip().split("\t", 1)
        status: dict[str, object] = {"state": output[0] if output[0] else "unknown"}
        if len(output) == 2 and output[1].isdigit():
            status["pid"] = int(output[1])
        return status

    def _stop_service(self, service: ServiceName) -> dict[str, object]:
        pid_path = posixpath.join(self.run_root, f"{service}.pid")
        module = SERVICE_MODULES[service]
        script = "\n".join(
            [
                "set -u",
                f"pid_file={shlex.quote(pid_path)}",
                "if [ ! -f \"$pid_file\" ]; then printf 'already-stopped\\n'; exit 0; fi",
                'pid=$(cat "$pid_file" 2>/dev/null || true)',
                (
                    "case \"$pid\" in ''|*[!0-9]*) printf '%s\\n' "
                    "'invalid PID file' >&2; exit 74 ;; esac"
                ),
                'if ! kill -0 "$pid" 2>/dev/null; then',
                "  rm -f \"$pid_file\"; printf 'stopped\\n'; exit 0",
                "fi",
                "cmdline=$(tr '\\000' ' ' </proc/\"$pid\"/cmdline 2>/dev/null || true)",
                f'case "$cmdline" in *{shlex.quote(module)}*) ;; *)',
                "  printf '%s\\n' 'PID belongs to a different process; refusing to signal it' >&2",
                "  exit 75",
                "esac",
                'kill -TERM "$pid"',
                "count=0",
                'while kill -0 "$pid" 2>/dev/null && [ "$count" -lt 100 ]; do',
                "  sleep 0.1; count=$((count + 1))",
                "done",
                'if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid"; fi',
                'rm -f "$pid_file"',
                "printf 'stopped\\n'",
            ]
        )
        state = self.transport.run(script).stdout.strip() or "stopped"
        return {"state": state}


__all__ = ["SERVICE_MODULES", "Go2ServiceSupervisor"]
