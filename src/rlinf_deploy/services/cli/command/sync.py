"""Synchronize local Deploy code without restarting inference services."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import posixpath
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from ...environment import EnvironmentProfile, UvEnvironmentManager
from ...executor import Command, Executor
from ...state import DeploymentState, NodeState, StateStore
from ..context import CommandContext
from ..parallel import run_on_nodes

_DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
_ROOT_FILES = (*_DEPENDENCY_FILES, "README.md")
_WORKER_PATTERN = "[r]linf_deploy.bindings.worker"


class SyncError(RuntimeError):
    """Local Deploy code cannot be synchronized safely."""


@dataclass(frozen=True, slots=True)
class SourceArchive:
    content: bytes
    digest: str
    dependency_digest: str
    dependency_files: dict[str, bytes]


@dataclass(frozen=True, slots=True)
class NodeSync:
    uploaded: bool
    dependencies_updated: bool


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "sync",
        help="synchronize local Deploy code without restarting inference",
    )
    parser.add_argument(
        "--target",
        choices=("deploy",),
        default="deploy",
        help="project to synchronize (default: deploy)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("."),
        help="local RLinf Deploy repository root (default: current directory)",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    source = _validate_source(args.source)
    archive = _build_archive(source)
    state = _initialized_state(context)
    profiles = _deploy_profiles_by_node(context, state)

    progress = context.progress
    progress.begin("sync", context.deployment.name)
    for node_id in profiles:
        progress.add_node(node_id, total=4)

    def synchronize(node_id: str) -> NodeSync:
        try:
            result = _sync_node(
                context,
                state,
                node_id,
                profiles[node_id],
                archive,
            )
        except Exception as error:
            progress.fail(node_id, error)
            raise
        progress.succeed(
            node_id,
            detail=(
                "Deploy overlay active"
                + ("; dependencies updated" if result.dependencies_updated else "")
            ),
        )
        return result

    results = run_on_nodes(profiles, synchronize)
    success = not results.errors
    progress.finish(success=success)
    if results.errors:
        raise SyncError(_error_summary(results.errors))

    uploads = sum(result.uploaded for result in results.values.values())
    progress.message(
        f"Deploy overlay {archive.digest[:12]} active on {len(results.values)} "
        f"node(s) ({uploads} upload(s)); inference services were not restarted"
    )
    return 0


def _initialized_state(context: CommandContext) -> DeploymentState:
    state = StateStore(context.state_path).load()
    if state is None:
        raise SyncError("deployment is not initialized; run `rlinf-deploy ... init`")
    return state


def _deploy_profiles_by_node(
    context: CommandContext,
    state: DeploymentState,
) -> dict[str, tuple[EnvironmentProfile, ...]]:
    profiles: dict[str, list[EnvironmentProfile]] = {}
    for profile in context.deployment.environments:
        if profile.project != "deploy":
            continue
        environment = state.environments.get(profile.environment_id)
        if environment is None or environment.status != "ready":
            raise SyncError(
                f"environment {profile.environment_id!r} is not ready; run init again"
            )
        if profile.node not in state.nodes:
            raise SyncError(f"initialized state is missing node {profile.node!r}")
        profiles.setdefault(profile.node, []).append(
            replace(profile, path=environment.path)
        )
    if not profiles:
        raise SyncError("deployment has no Deploy environments to synchronize")
    return {
        node_id: tuple(node_profiles)
        for node_id, node_profiles in sorted(profiles.items())
    }


def _sync_node(
    context: CommandContext,
    state: DeploymentState,
    node_id: str,
    profiles: tuple[EnvironmentProfile, ...],
    archive: SourceArchive,
) -> NodeSync:
    progress = context.progress
    node = state.nodes[node_id]
    overlay_root = posixpath.join(node.root, "overlays", "deploy")
    release = posixpath.join(overlay_root, "releases", archive.digest)
    archive_path = posixpath.join(
        overlay_root,
        "archives",
        f"{archive.digest}.tar.gz",
    )
    complete_marker = posixpath.join(release, ".complete")
    dependency_marker = posixpath.join(overlay_root, "dependency-digest")
    current = posixpath.join(overlay_root, "current")

    with context.executor(node_id) as executor:
        progress.update(node_id, "Checking for active robot workers")
        _require_idle_worker(executor)
        progress.advance(node_id)

        progress.update(node_id, "Uploading Deploy source", detail=archive.digest[:12])
        uploaded = _install_release(
            executor,
            overlay_root=overlay_root,
            release=release,
            archive_path=archive_path,
            complete_marker=complete_marker,
            archive=archive,
        )
        progress.advance(node_id)

        progress.update(node_id, "Checking Deploy dependencies")
        dependencies_updated = _synchronize_dependencies(
            executor,
            node=node,
            release=release,
            dependency_marker=dependency_marker,
            profiles=profiles,
            archive=archive,
        )
        progress.advance(node_id)

        progress.update(node_id, "Activating Deploy overlay")
        executor.replace_symlink(current, release)
        progress.advance(node_id)
    return NodeSync(uploaded, dependencies_updated)


def _require_idle_worker(executor: Executor) -> None:
    result = executor.run(
        Command(("pgrep", "-f", _WORKER_PATTERN)),
        check=False,
    )
    if result.exit_code == 0:
        processes = ", ".join(result.stdout.split()) or "unknown PID"
        raise SyncError(
            "a robot binding worker is running "
            f"({processes}); wait for `run` to finish before syncing"
        )
    if result.exit_code != 1:
        detail = result.stderr.strip() or f"exit code {result.exit_code}"
        raise SyncError(f"could not check robot binding workers: {detail}")


def _install_release(
    executor: Executor,
    *,
    overlay_root: str,
    release: str,
    archive_path: str,
    complete_marker: str,
    archive: SourceArchive,
) -> bool:
    existing = executor.run(Command(("test", "-f", complete_marker)), check=False)
    if existing.exit_code == 0:
        return False
    if existing.exit_code != 1:
        detail = existing.stderr.strip() or f"exit code {existing.exit_code}"
        raise SyncError(f"could not inspect Deploy overlay: {detail}")

    executor.run(
        Command(
            (
                "mkdir",
                "-p",
                posixpath.join(overlay_root, "archives"),
                release,
            )
        )
    )
    executor.write_bytes(archive_path, archive.content, mode=0o600)
    executor.run(Command(("tar", "-xzf", archive_path, "-C", release)))
    executor.write_text(complete_marker, f"{archive.digest}\n", mode=0o600)
    return True


def _synchronize_dependencies(
    executor: Executor,
    *,
    node: NodeState,
    release: str,
    dependency_marker: str,
    profiles: tuple[EnvironmentProfile, ...],
    archive: SourceArchive,
) -> bool:
    recorded = _read_optional_file(executor, dependency_marker)
    expected = f"{archive.dependency_digest}\n".encode()
    if recorded == expected:
        return False

    changed = not _managed_dependencies_match(executor, node, archive)
    if changed:
        uv = _uv_executable(executor, node)
        manager = UvEnvironmentManager(executor, uv_executable=uv)
        for profile in profiles:
            manager.prepare(profile, project_dir=release)

    executor.write_text(
        dependency_marker,
        expected.decode(),
        mode=0o600,
    )
    return changed


def _managed_dependencies_match(
    executor: Executor,
    node: NodeState,
    archive: SourceArchive,
) -> bool:
    for name, expected in archive.dependency_files.items():
        remote_path = posixpath.join(node.deploy_project, name)
        available = executor.run(Command(("test", "-f", remote_path)), check=False)
        if available.exit_code == 1:
            return False
        if available.exit_code != 0:
            detail = available.stderr.strip() or f"exit code {available.exit_code}"
            raise SyncError(f"could not inspect {remote_path}: {detail}")
        if executor.read_bytes(remote_path) != expected:
            return False
    return True


def _read_optional_file(executor: Executor, path: str) -> bytes | None:
    available = executor.run(Command(("test", "-f", path)), check=False)
    if available.exit_code == 1:
        return None
    if available.exit_code != 0:
        detail = available.stderr.strip() or f"exit code {available.exit_code}"
        raise SyncError(f"could not inspect {path}: {detail}")
    return executor.read_bytes(path)


def _uv_executable(executor: Executor, node: NodeState) -> str:
    candidate = posixpath.join(node.home, ".local", "bin", "uv")
    available = executor.run(Command(("test", "-x", candidate)), check=False)
    if available.exit_code == 0:
        return candidate
    if available.exit_code == 1:
        return "uv"
    detail = available.stderr.strip() or f"exit code {available.exit_code}"
    raise SyncError(f"could not inspect uv executable: {detail}")


def _validate_source(value: Path) -> Path:
    source = value.expanduser().resolve()
    missing = [
        name
        for name in (*_ROOT_FILES, "src/rlinf_deploy")
        if not (source / name).exists()
    ]
    if missing:
        raise SyncError(
            f"{source} is not an RLinf Deploy source tree; missing: "
            + ", ".join(missing)
        )
    if not (source / "src" / "rlinf_deploy").is_dir():
        raise SyncError(f"{source / 'src/rlinf_deploy'} is not a directory")
    return source


def _build_archive(source: Path) -> SourceArchive:
    files = {
        name: (source / name).read_bytes()
        for name in _ROOT_FILES
    }
    package_root = source / "src" / "rlinf_deploy"
    for path in sorted(package_root.rglob("*")):
        relative_parts = path.relative_to(source).parts
        if path.is_symlink() or not path.is_file() or _excluded(relative_parts):
            continue
        files[PurePosixPath(*relative_parts).as_posix()] = path.read_bytes()

    digest = _content_digest(files)
    dependency_files = {name: files[name] for name in _DEPENDENCY_FILES}
    dependency_digest = _content_digest(dependency_files)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            directories = {
                parent.as_posix()
                for name in files
                for parent in PurePosixPath(name).parents
                if parent != PurePosixPath(".")
            }
            for name in sorted(directories, key=lambda item: (item.count("/"), item)):
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                _normalize_tar_info(info)
                archive.addfile(info)
            for name, content in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                _normalize_tar_info(info)
                archive.addfile(info, io.BytesIO(content))
    return SourceArchive(output.getvalue(), digest, dependency_digest, dependency_files)


def _excluded(parts: tuple[str, ...]) -> bool:
    return "__pycache__" in parts or any(
        part.endswith((".pyc", ".pyo")) for part in parts
    )


def _normalize_tar_info(info: tarfile.TarInfo) -> None:
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""


def _content_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        encoded_name = name.encode()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _error_summary(errors: dict[str, Exception]) -> str:
    return "; ".join(
        f"{node}: {error}" for node, error in sorted(errors.items())
    )


__all__ = ["SyncError", "register", "run"]
