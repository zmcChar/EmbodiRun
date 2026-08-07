from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from embodied_runtime.robots.unitree.go2.agent.control import (
    ensure_cyclonedds_library_dir,
)


def test_cyclonedds_validation_rejects_missing_and_unsafe_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        ensure_cyclonedds_library_dir(str(tmp_path))

    library = tmp_path / "libddsc.so.0"
    library.write_bytes(b"ELF\x00iox_pub_publish_chunk\x00")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="incompatible Iceoryx"):
        ensure_cyclonedds_library_dir(str(tmp_path))


def test_cyclonedds_reexec_uses_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "libddsc.so.0"
    library.write_bytes(b"safe-placeholder")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere/else")
    captured: dict[str, Any] = {}

    class ExecCalled(Exception):
        pass

    def fake_execve(executable: str, argv: list[str], environ: dict[str, str]) -> None:
        captured.update(executable=executable, argv=argv, environ=environ)
        raise ExecCalled

    monkeypatch.setattr(os, "execve", fake_execve)

    with pytest.raises(ExecCalled):
        ensure_cyclonedds_library_dir(
            str(tmp_path),
            reexec_args=["--mode", "live"],
        )

    assert captured["executable"] == sys.executable
    assert captured["argv"] == [
        sys.executable,
        "-m",
        "embodied_runtime.robots.unitree.go2.agent.control",
        "--mode",
        "live",
    ]
    assert captured["environ"]["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(tmp_path.resolve())
