from __future__ import annotations

import argparse
from typing import Any

import pytest

from embodied_runtime.robots.unitree.go2.agent.control import (
    ActionExecutor,
    ControlServerConfig,
    RobotState,
    build_parser,
    config_from_args,
)
from embodied_runtime.robots.unitree.go2.agent.control import cli as control_cli


def test_cli_resolves_every_deployment_setting_from_environment() -> None:
    token = "z" * 32
    parser = build_parser(
        {
            "GO2_CONTROL_MODE": "live",
            "GO2_CONTROL_HOST": "192.0.2.10",
            "GO2_CONTROL_PORT": "18080",
            "GO2_DDS_INTERFACE": "robot0",
            "GO2_STATE_TOPIC": "rt/custom/state",
            "GO2_CYCLONEDDS_LIB_DIR": "/opt/cyclonedds/lib",
            "GO2_API_TOKEN": token,
            "GO2_OPERATOR_READY": "yes",
            "GO2_SDK_RPC_TIMEOUT": "3.5",
        }
    )
    config = config_from_args(parser.parse_args([]))

    assert config == ControlServerConfig(
        mode="live",
        host="192.0.2.10",
        port=18080,
        interface="robot0",
        state_topic="rt/custom/state",
        cyclonedds_lib_dir="/opt/cyclonedds/lib",
        token=token,
        operator_ready=True,
        rpc_timeout_s=3.5,
    )


def test_cli_arguments_override_environment() -> None:
    parser = build_parser(
        {
            "GO2_CONTROL_HOST": "127.0.0.2",
            "GO2_CONTROL_PORT": "8081",
            "GO2_OPERATOR_READY": "true",
        }
    )
    config = config_from_args(
        parser.parse_args(
            [
                "--host",
                "localhost",
                "--port",
                "9090",
                "--no-operator-ready",
                "--token",
                "cli-token",
            ]
        )
    )

    assert config.host == "localhost"
    assert config.port == 9090
    assert config.operator_ready is False
    assert config.token == "cli-token"


def test_cli_boolean_flags_do_not_require_boolean_optional_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 3.8 has no argparse.BooleanOptionalAction."""

    monkeypatch.delattr(argparse, "BooleanOptionalAction", raising=False)
    parser = build_parser({"GO2_OPERATOR_READY": "true"})

    assert parser.parse_args([]).operator_ready is True
    assert parser.parse_args(["--no-operator-ready"]).operator_ready is False
    assert parser.parse_args(["--operator-ready"]).operator_ready is True


def test_runtime_dataclasses_do_not_require_python310_slots_argument() -> None:
    config = ControlServerConfig(
        mode="dry-run",
        host="127.0.0.1",
        port=8080,
        interface=None,
        state_topic="rt/lf/sportmodestate",
        cyclonedds_lib_dir=None,
        token=None,
        operator_ready=False,
    )
    state = RobotState(
        received_at=0.0,
        sequence=1,
        position=(0.0, 0.0, 0.0),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        velocity=(0.0, 0.0, 0.0),
        yaw_rate=0.0,
    )

    assert hasattr(config, "__dict__")
    assert hasattr(state, "__dict__")


def test_live_config_requires_explicit_interface_and_library() -> None:
    parser = build_parser({})
    with pytest.raises(ValueError, match="DDS network interface"):
        config_from_args(parser.parse_args(["--mode", "live"]))


def test_main_accepts_argv_and_builds_dry_run_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_serve(config: ControlServerConfig, executor: ActionExecutor) -> None:
        captured.update(config=config, snapshot=executor.snapshot())
        executor.close()

    monkeypatch.setattr(control_cli, "serve_control_api", fake_serve)

    status = control_cli.main(
        [
            "--mode",
            "dry-run",
            "--host",
            "127.0.0.1",
            "--port",
            "19090",
        ]
    )

    assert status == 0
    assert captured["config"].port == 19090
    assert captured["snapshot"]["transport"] == "dry-run"
    assert captured["snapshot"]["operator_motion_ready"] is True


def test_main_rejects_external_bind_without_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    def fake_serve(config: ControlServerConfig, executor: ActionExecutor) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(control_cli, "serve_control_api", fake_serve)

    assert control_cli.main(["--host", "0.0.0.0"]) == 2
    assert not called
    assert "GO2_API_TOKEN" in capsys.readouterr().err
