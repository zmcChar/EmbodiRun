"""Install and supervise the repository-owned Go2 edge services."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import posixpath
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from .transport import RemoteTransport

ServiceName = Literal["control", "camera"]
ServiceSelection = Literal["all", "control", "camera"]

_SERVICE_MODULES: dict[ServiceName, str] = {
    "control": "embodied_runtime.robots.go2.agent.control",
    "camera": "embodied_runtime.robots.go2.agent.camera",
}
_IGNORED_DIRECTORY_NAMES = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
_IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class ControlStartOptions:
    """Remote control-service arguments."""

    mode: Literal["dry-run", "live"] = "live"
    interface: str = "eno2"
    state_topic: str = "rt/lf/sportmodestate"
    cyclonedds_lib_dir: str | None = None
    bind: str = "127.0.0.1"
    port: int = 8080
    api_token: str | None = field(default=None, repr=False)
    operator_ready: bool = False

    def __post_init__(self) -> None:
        _validate_port(self.port, "control")
        _validate_token_for_bind(self.bind, self.api_token, "control API")
        if self.mode == "live" and not self.cyclonedds_lib_dir:
            raise ValueError("live control requires --cyclonedds-lib-dir")


@dataclass(frozen=True)
class CameraStartOptions:
    """Remote camera-service arguments."""

    backend: Literal["realsense", "v4l2"] = "realsense"
    realsense_serial: str | None = None
    depth_scale: float | None = None
    bind: str = "127.0.0.1"
    port: int = 8765
    camera_token: str | None = field(default=None, repr=False)
    width: int = 640
    height: int = 360
    camera_fps: int = 15
    jpeg_fps: float = 5.0
    device: str = "/dev/video4"

    def __post_init__(self) -> None:
        _validate_port(self.port, "camera")
        _validate_token_for_bind(self.bind, self.camera_token, "camera API")
        if self.width <= 0 or self.height <= 0 or self.width % 2:
            raise ValueError("camera dimensions must be positive and width must be even")
        if self.camera_fps <= 0 or self.jpeg_fps <= 0:
            raise ValueError("camera frame rates must be positive")
        if self.backend == "realsense" and (self.depth_scale is None or self.depth_scale <= 0):
            raise ValueError("RealSense camera requires an explicit positive --depth-scale")


@dataclass(frozen=True)
class StartOptions:
    """Complete start request for one or both edge services."""

    python: str
    services: ServiceSelection = "all"
    control: ControlStartOptions | None = None
    camera: CameraStartOptions | None = None

    def __post_init__(self) -> None:
        if not self.python.strip():
            raise ValueError("remote Python path cannot be empty")
        if self.services in {"all", "control"} and self.control is None:
            raise ValueError("control options are required for the selected services")
        if self.services in {"all", "camera"} and self.camera is None:
            raise ValueError("camera options are required for the selected services")


class Go2AgentDeployer:
    """High-level deployment operations independent of the SSH implementation."""

    def __init__(self, transport: RemoteTransport, remote_root: str) -> None:
        self.transport = transport
        self.remote_root = transport.resolve_path(remote_root)
        if self.remote_root == "/":
            raise ValueError("remote root cannot be the filesystem root")
        self.source_root = posixpath.join(self.remote_root, "src")
        self.run_root = posixpath.join(self.remote_root, "run")
        self.log_root = posixpath.join(self.remote_root, "logs")

    def probe(self, python: str = "python3") -> dict[str, object]:
        """Verify SSH command execution and report the remote Python platform."""

        probe_code = (
            "import json,platform,sys;"
            "print(json.dumps({'python':sys.executable,'python_version':platform.python_version(),"
            "'machine':platform.machine(),'platform':platform.system()}))"
        )
        result = self.transport.run(shlex.join([python, "-c", probe_code]))
        try:
            payload = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError("remote Python probe returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("remote Python probe returned an invalid payload")
        return cast(dict[str, object], payload)

    def install(self, package_root: Path | None = None) -> dict[str, object]:
        """Upload the current ``embodied_runtime`` package through SFTP."""

        local_root = (package_root or _default_package_root()).resolve()
        _validate_package_root(local_root)
        files = list(_iter_deployment_files(local_root))
        if not files:
            raise RuntimeError(f"no deployable files found below {local_root}")

        remote_package = posixpath.join(self.source_root, "embodied_runtime")
        self.transport.make_dirs(remote_package)
        self.transport.make_dirs(self.run_root, mode=0o700)
        self.transport.make_dirs(self.log_root, mode=0o700)

        digest = hashlib.sha256()
        for local_path in files:
            relative = local_path.relative_to(local_root).as_posix()
            payload = local_path.read_bytes()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload)
            remote_path = posixpath.join(remote_package, relative)
            self.transport.put_file(local_path, remote_path)

        manifest = {
            "format": 1,
            "package": "embodied_runtime",
            "file_count": len(files),
            "sha256": digest.hexdigest(),
            "source_root": self.source_root,
            "modules": _SERVICE_MODULES,
        }
        manifest_path = posixpath.join(self.remote_root, "deployment-manifest.json")
        self.transport.put_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            manifest_path,
            mode=0o644,
        )
        return manifest

    def start(self, options: StartOptions) -> dict[str, object]:
        """Start selected services as independent, supervised-by-PID processes."""

        results: dict[str, object] = {}
        for service in _selected_services(options.services):
            if service == "control":
                assert options.control is not None
                command, environment = self._control_command(options.python, options.control)
            else:
                assert options.camera is not None
                command, environment = self._camera_command(options.python, options.camera)
            results[service] = self._start_service(service, command, environment)
        return {"services": results}

    def status(self, services: ServiceSelection = "all") -> dict[str, object]:
        """Return PID-backed status without changing remote state."""

        statuses = {
            service: self._status_service(service) for service in _selected_services(services)
        }
        return {"services": statuses}

    def stop(self, services: ServiceSelection = "all") -> dict[str, object]:
        """Gracefully stop selected services, escalating only after a timeout."""

        # Stop the control service first so its signal handler issues StopMove
        # before camera teardown or SSH disconnects.
        selected = _selected_services(services)
        selected.sort(key=lambda name: 0 if name == "control" else 1)
        stopped = {service: self._stop_service(service) for service in selected}
        return {"services": stopped}

    def _control_command(
        self,
        python: str,
        options: ControlStartOptions,
    ) -> tuple[list[str], dict[str, str]]:
        command = [
            python,
            "-m",
            _SERVICE_MODULES["control"],
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
            _SERVICE_MODULES["camera"],
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
        module = _SERVICE_MODULES[service]
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
        module = _SERVICE_MODULES[service]
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
        module = _SERVICE_MODULES[service]
        script = "\n".join(
            [
                "set -u",
                f"pid_file={shlex.quote(pid_path)}",
                "if [ ! -f \"$pid_file\" ]; then printf 'already-stopped\\n'; exit 0; fi",
                'pid=$(cat "$pid_file" 2>/dev/null || true)',
                "case \"$pid\" in ''|*[!0-9]*) printf '%s\\n' 'invalid PID file' >&2; exit 74 ;; esac",
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


def _selected_services(selection: ServiceSelection) -> list[ServiceName]:
    if selection == "all":
        return ["control", "camera"]
    return [cast(ServiceName, selection)]


def _default_package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_package_root(path: Path) -> None:
    if path.name != "embodied_runtime" or not (path / "__init__.py").is_file():
        raise ValueError(f"package root is not an embodied_runtime source tree: {path}")
    for module in _SERVICE_MODULES.values():
        relative = Path(*module.split(".")[1:])
        if not ((path / relative).is_dir() or (path / relative).with_suffix(".py").is_file()):
            raise ValueError(f"dog-side agent module is missing: {module}")


def _iter_deployment_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if any(part in _IGNORED_DIRECTORY_NAMES for part in relative_parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix in _IGNORED_FILE_SUFFIXES or path.name == ".DS_Store":
            continue
        yield path


def _validate_port(port: int, label: str) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"{label} port must be between 1 and 65535")


def _validate_token_for_bind(bind: str, token: str | None, label: str) -> None:
    try:
        loopback = bind.strip().lower() == "localhost" or ipaddress.ip_address(bind).is_loopback
    except ValueError:
        loopback = False
    if not loopback and token is None:
        raise ValueError(f"{label} token is required when binding beyond localhost")
    if token is not None:
        if len(token) < 32:
            raise ValueError(f"{label} token must contain at least 32 characters")
        if not token.isascii() or token.strip() != token or any(char.isspace() for char in token):
            raise ValueError(f"{label} token must be ASCII without whitespace")
