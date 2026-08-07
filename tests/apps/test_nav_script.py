from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NAV_SCRIPT = REPOSITORY_ROOT / "nav.sh"


def _without_machine_defaults() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "GO2_ROBOT_HOST",
        "GO2_SSH_USER",
        "GO2_CAMERA_SERIAL",
        "GO2_DEPTH_SCALE",
    ):
        environment.pop(name, None)
    return environment


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--prompt", "find target"], "--robot-host is required"),
        (
            ["--prompt", "find target", "--robot-host", "go2.example"],
            "--ssh-user is required",
        ),
        (
            [
                "--prompt",
                "find target",
                "--robot-host",
                "go2.example",
                "--ssh-user",
                "operator",
            ],
            "--camera-serial is required",
        ),
        (
            [
                "--prompt",
                "find target",
                "--robot-host",
                "go2.example",
                "--ssh-user",
                "operator",
                "--camera-serial",
                "camera-1",
            ],
            "--depth-scale must be a positive finite decimal",
        ),
    ],
)
def test_machine_specific_values_are_required(arguments: list[str], message: str) -> None:
    completed = subprocess.run(
        ["bash", os.fspath(NAV_SCRIPT), *arguments],
        cwd=REPOSITORY_ROOT,
        env=_without_machine_defaults(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def _write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.parametrize(
    ("backend_arguments", "root_environment", "model_environment", "root_option"),
    [
        ((), "STREAMVLN_ROOT", "STREAMVLN_MODEL_PATH", "--streamvln-root"),
        (("--backend", "navila"), "NAVILA_ROOT", "NAVILA_MODEL_PATH", "--navila-root"),
    ],
)
def test_one_command_stops_installs_restarts_then_runs_navigation(
    tmp_path: Path,
    backend_arguments: tuple[str, ...],
    root_environment: str,
    model_environment: str,
    root_option: str,
) -> None:
    log_path = tmp_path / "commands.jsonl"
    fake_deploy_python = tmp_path / "fake-deploy-python"
    _write_executable(
        fake_deploy_python,
        """
        #!/usr/bin/env python3
        import json
        import os
        import sys

        with open(os.environ["NAV_TEST_LOG"], "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": "deploy", "args": sys.argv[1:]}) + "\\n")
        """,
    )
    fake_navigation_python = tmp_path / "fake-navigation-python"
    _write_executable(
        fake_navigation_python,
        """
        #!/usr/bin/env python3
        import json
        import os
        import sys

        with open(os.environ["NAV_TEST_LOG"], "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": "navigation", "args": sys.argv[1:]}) + "\\n")
        """,
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "curl",
        """
        #!/usr/bin/env bash
        printf '%s' '{"operator_motion_ready":true}'
        """,
    )

    secrets_file = tmp_path / "go2.env"
    secrets_file.write_text(
        "GO2_API_TOKEN=" + "a" * 32 + "\n" + "GO2_CAMERA_TOKEN=" + "c" * 32 + "\n",
        encoding="utf-8",
    )
    model_root = tmp_path / "model-source"
    checkpoint = tmp_path / "checkpoint"
    model_root.mkdir()
    checkpoint.mkdir()

    environment = _without_machine_defaults()
    environment.update(
        {
            "GO2_DEPLOY_PYTHON": os.fspath(fake_deploy_python),
            "GO2_NAV_PYTHON": os.fspath(fake_navigation_python),
            "GO2_NAV_SECRETS_FILE": os.fspath(secrets_file),
            "GO2_SSH_PASSWORD": "ssh-password",
            "NAV_TEST_LOG": os.fspath(log_path),
            "PATH": os.fspath(fake_bin) + os.pathsep + environment["PATH"],
            root_environment: os.fspath(model_root),
            model_environment: os.fspath(checkpoint),
        }
    )
    completed = subprocess.run(
        [
            "bash",
            os.fspath(NAV_SCRIPT),
            *backend_arguments,
            "--prompt",
            "find the tripod",
            "--robot-host",
            "go2.example",
            "--ssh-user",
            "operator",
            "--camera-serial",
            "camera-1",
            "--depth-scale",
            "0.001",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    deploy_arguments = [record["args"] for record in records if record["kind"] == "deploy"]
    commands = [
        next(command for command in ("stop", "install", "start") if command in args)
        for args in deploy_arguments
    ]
    assert commands == ["stop", "install", "start", "start"]

    stop_arguments, install_arguments, camera_arguments, control_arguments = deploy_arguments
    assert stop_arguments[-3:] == ["stop", "--service", "all"]
    assert install_arguments[-3:] == [
        "install",
        "--package-root",
        os.fspath(REPOSITORY_ROOT / "src/embodied_runtime"),
    ]
    assert camera_arguments[camera_arguments.index("--service") + 1] == "camera"
    assert camera_arguments[camera_arguments.index("--realsense-serial") + 1] == "camera-1"
    assert camera_arguments[camera_arguments.index("--depth-scale") + 1] == "0.001"
    assert control_arguments[control_arguments.index("--service") + 1] == "control"
    assert "--operator-ready" in control_arguments

    navigation = next(record["args"] for record in records if record["kind"] == "navigation")
    assert navigation[:3] == ["-m", "embodied_runtime.apps.navigate", "--config"]
    expected_backend = "navila" if backend_arguments else "streamvln"
    assert navigation[navigation.index("--backend") + 1] == expected_backend
    assert navigation[navigation.index(root_option) + 1] == os.fspath(model_root)
    assert navigation[navigation.index("--robot-host") + 1] == "go2.example"
    assert "--execute" in navigation
