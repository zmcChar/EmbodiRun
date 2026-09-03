"""Local deployment command execution."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path

from .command import Command, CommandError, CommandResult
from .response import JsonHttpResponse

_MAX_JSON_RESPONSE_BYTES = 64 * 1024


class LocalExecutor:
    """Execute commands directly without a shell."""

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        environment = os.environ.copy()
        environment.update(command.environment)
        try:
            completed = subprocess.run(
                command.argv,
                cwd=command.cwd,
                env=environment,
                input=command.stdin,
                capture_output=True,
                text=True,
                timeout=command.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("local command timed out") from error
        except OSError as error:
            raise RuntimeError(f"local command could not start: {error}") from error
        result = CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and result.exit_code != 0:
            raise CommandError(result.exit_code, result.stderr)
        return result

    def get_json(self, url: str, *, timeout_s: float) -> JsonHttpResponse:
        """Read JSON from an HTTP endpoint reachable by the local node."""

        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read(_MAX_JSON_RESPONSE_BYTES + 1)
            status = int(response.status)
        if len(body) > _MAX_JSON_RESPONSE_BYTES:
            raise ValueError("HTTP JSON response is too large")
        return JsonHttpResponse(status, json.loads(body))

    def write_text(self, path: str, content: str, *, mode: int = 0o600) -> None:
        """Atomically write a UTF-8 text file for a local deployment node."""

        self.write_bytes(path, content.encode("utf-8"), mode=mode)

    def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def replace_symlink(self, path: str, target: str) -> None:
        """Atomically point one local symlink at a new target."""

        link = Path(path)
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.symlink_to(target)
            os.replace(temporary, link)
        finally:
            temporary.unlink(missing_ok=True)

    def write_bytes(
        self,
        path: str,
        content: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        """Atomically write bytes for a local deployment node."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        return None


__all__ = ["LocalExecutor"]
