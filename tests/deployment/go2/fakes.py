from __future__ import annotations

import posixpath
from pathlib import Path

from embodied_runtime.deployment.go2.transport import CommandResult


class FakeTransport:
    def __init__(self, results: list[CommandResult] | None = None) -> None:
        self.results = list(results or [])
        self.commands: list[tuple[str, bool]] = []
        self.directories: list[tuple[str, int]] = []
        self.files: dict[str, bytes] = {}
        self.modes: dict[str, int] = {}
        self.closed = False

    def run(self, command: str, *, check: bool = True) -> CommandResult:
        self.commands.append((command, check))
        if self.results:
            return self.results.pop(0)
        return CommandResult(0, "")

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join("/home/tester", path.removeprefix("~/")))

    def make_dirs(self, path: str, *, mode: int = 0o755) -> None:
        self.directories.append((path, mode))

    def put_file(self, local_path: Path, remote_path: str, *, mode: int = 0o644) -> None:
        self.files[remote_path] = local_path.read_bytes()
        self.modes[remote_path] = mode

    def put_bytes(self, payload: bytes, remote_path: str, *, mode: int = 0o600) -> None:
        self.files[remote_path] = payload
        self.modes[remote_path] = mode

    def close(self) -> None:
        self.closed = True
