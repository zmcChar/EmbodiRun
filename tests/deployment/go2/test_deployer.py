from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_runtime.deployment.go2.deployer import (
    CameraStartOptions,
    ControlStartOptions,
    Go2AgentDeployer,
    StartOptions,
)
from embodied_runtime.deployment.go2.transport import CommandResult

from .fakes import FakeTransport


def _package_tree(root: Path) -> Path:
    package = root / "embodied_runtime"
    (package / "robots/go2/agent/control").mkdir(parents=True)
    (package / "robots/go2/agent/camera").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "robots/go2/agent/control/__main__.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    (package / "robots/go2/agent/camera/__main__.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    cache = package / "robots/go2/agent/control/__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"cache")
    return package


def test_install_uploads_repository_package_and_manifest(tmp_path: Path) -> None:
    package = _package_tree(tmp_path)
    transport = FakeTransport()
    deployer = Go2AgentDeployer(transport, "deploy/go2")

    manifest = deployer.install(package)

    prefix = "/home/tester/deploy/go2/src/embodied_runtime"
    assert transport.files[f"{prefix}/__init__.py"] == b""
    assert transport.files[f"{prefix}/robots/go2/agent/control/__main__.py"]
    assert all("__pycache__" not in path for path in transport.files)
    remote_manifest = json.loads(
        transport.files["/home/tester/deploy/go2/deployment-manifest.json"]
    )
    assert remote_manifest == manifest
    assert manifest["file_count"] == 3


def test_start_builds_separate_quoted_commands_and_secret_env_files() -> None:
    transport = FakeTransport([CommandResult(0, "1201\n"), CommandResult(0, "1202\n")])
    deployer = Go2AgentDeployer(transport, "/opt/rlinf go2")
    api_token = "api_" + "a" * 32
    camera_token = "camera_" + "b" * 32

    result = deployer.start(
        StartOptions(
            python="/opt/unitree env/bin/python",
            control=ControlStartOptions(
                interface="enp 2",
                state_topic="rt/topic;false",
                cyclonedds_lib_dir="/opt/dds libs",
                bind="0.0.0.0",
                api_token=api_token,
            ),
            camera=CameraStartOptions(
                realsense_serial="serial;false",
                depth_scale=0.001,
                bind="0.0.0.0",
                camera_token=camera_token,
            ),
        )
    )

    assert result["services"]["control"]["pid"] == 1201
    assert result["services"]["camera"]["pid"] == 1202
    control_command, camera_command = [entry[0] for entry in transport.commands]
    assert "--operator-ready" not in control_command
    assert "'enp 2'" in control_command
    assert "'rt/topic;false'" in control_command
    assert "'serial;false'" in camera_command
    assert api_token not in control_command
    assert camera_token not in camera_command
    assert api_token.encode() in transport.files["/opt/rlinf go2/run/control.env"]
    assert camera_token.encode() in transport.files["/opt/rlinf go2/run/camera.env"]
    assert transport.modes["/opt/rlinf go2/run/control.env"] == 0o600


def test_operator_ready_is_only_forwarded_when_explicit() -> None:
    transport = FakeTransport([CommandResult(0, "99\n")])
    deployer = Go2AgentDeployer(transport, "/opt/go2")

    deployer.start(
        StartOptions(
            python="python3",
            services="control",
            control=ControlStartOptions(
                cyclonedds_lib_dir="/opt/dds",
                operator_ready=True,
            ),
        )
    )

    assert "--operator-ready" in transport.commands[0][0]


def test_realsense_requires_explicit_calibrated_depth_scale() -> None:
    with pytest.raises(ValueError, match="depth-scale"):
        CameraStartOptions()


def test_non_loopback_bind_requires_token() -> None:
    with pytest.raises(ValueError, match="token"):
        ControlStartOptions(cyclonedds_lib_dir="/opt/dds", bind="0.0.0.0")


def test_status_and_stop_verify_service_module_before_signalling() -> None:
    transport = FakeTransport([CommandResult(0, "running\t123\n"), CommandResult(0, "stopped\n")])
    deployer = Go2AgentDeployer(transport, "/opt/go2")

    assert deployer.status("control") == {"services": {"control": {"state": "running", "pid": 123}}}
    assert deployer.stop("control") == {"services": {"control": {"state": "stopped"}}}

    status_command, stop_command = [entry[0] for entry in transport.commands]
    module = "embodied_runtime.robots.go2.agent.control"
    assert module in status_command
    assert module in stop_command
    assert "kill -TERM" in stop_command


def test_probe_parses_remote_platform_json() -> None:
    payload = {"python": "/usr/bin/python3", "python_version": "3.10.12"}
    transport = FakeTransport([CommandResult(0, json.dumps(payload))])

    assert Go2AgentDeployer(transport, "/opt/go2").probe() == payload
