from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from embodied_runtime.robots.unitree.go2 import Go2ControlClient
from embodied_runtime.robots.unitree.go2.client import (
    Go2ControlClient as ModuleGo2ControlClient,
)


def test_go2_root_exports_hardware_clients_without_task_value_aliases() -> None:
    assert Go2ControlClient is ModuleGo2ControlClient
    go2_root = Path(__file__).resolve().parents[4] / "src/embodied_runtime/robots/unitree/go2"
    assert all(not (go2_root / name).exists() for name in ("limits.py", "motion.py", "state.py"))


def test_agent_sources_parse_as_python38_and_import_without_host_tasks() -> None:
    source_root = Path(__file__).resolve().parents[4] / "src"
    agent_root = source_root / "embodied_runtime/robots/unitree/go2/agent"
    paths = tuple(agent_root.rglob("*.py"))
    assert paths
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(source_root)
    code = (
        "import sys;"
        "import embodied_runtime.robots.unitree.go2.agent.control;"
        "import embodied_runtime.robots.unitree.go2.agent.camera;"
        "assert not any(n.startswith('embodied_runtime.tasks') for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_agent_cli_module_paths_expose_help_without_hardware_imports() -> None:
    source_root = Path(__file__).resolve().parents[4] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(source_root)
    expected_options = {
        "control": ("--mode", "--interface", "--state-topic"),
        "camera": ("--backend", "--width", "--height"),
    }
    for service, options in expected_options.items():
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                f"embodied_runtime.robots.unitree.go2.agent.{service}",
                "--help",
            ],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert all(option in completed.stdout for option in options)
