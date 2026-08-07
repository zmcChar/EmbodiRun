"""Install/probe facade for the repository-owned Unitree Go2 edge agent."""

from __future__ import annotations

import hashlib
import json
import posixpath
import shlex
from pathlib import Path
from typing import cast

from .options import ServiceSelection, StartOptions
from .services import SERVICE_MODULES, Go2ServiceSupervisor
from .transport import RemoteTransport

_IGNORED_DIRECTORY_NAMES = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
_IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}


class Go2AgentDeployer:
    """Install the agent and delegate its lifecycle to a service supervisor."""

    def __init__(self, transport: RemoteTransport, remote_root: str) -> None:
        self.transport = transport
        self.remote_root = transport.resolve_path(remote_root)
        if self.remote_root == "/":
            raise ValueError("remote root cannot be the filesystem root")
        self.source_root = posixpath.join(self.remote_root, "src")
        self.run_root = posixpath.join(self.remote_root, "run")
        self.log_root = posixpath.join(self.remote_root, "logs")
        self.services = Go2ServiceSupervisor(
            transport,
            source_root=self.source_root,
            run_root=self.run_root,
            log_root=self.log_root,
        )

    def probe(self, python: str = "python3") -> dict[str, object]:
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
            self.transport.put_file(local_path, posixpath.join(remote_package, relative))

        manifest = {
            "format": 1,
            "package": "embodied_runtime",
            "file_count": len(files),
            "sha256": digest.hexdigest(),
            "source_root": self.source_root,
            "modules": SERVICE_MODULES,
        }
        self.transport.put_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            posixpath.join(self.remote_root, "deployment-manifest.json"),
            mode=0o644,
        )
        return manifest

    def start(self, options: StartOptions) -> dict[str, object]:
        return self.services.start(options)

    def status(self, services: ServiceSelection = "all") -> dict[str, object]:
        return self.services.status(services)

    def stop(self, services: ServiceSelection = "all") -> dict[str, object]:
        return self.services.stop(services)


def _default_package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_package_root(path: Path) -> None:
    if path.name != "embodied_runtime" or not (path / "__init__.py").is_file():
        raise ValueError(f"package root is not an embodied_runtime source tree: {path}")
    for module in SERVICE_MODULES.values():
        if not _module_exists(path, module):
            raise ValueError(f"dog-side agent module is missing: {module}")


def _module_exists(package_root: Path, module: str) -> bool:
    relative = Path(*module.split(".")[1:])
    return (package_root / relative).is_dir() or (package_root / relative).with_suffix(
        ".py"
    ).is_file()


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


__all__ = ["Go2AgentDeployer"]
