from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from embodied_runtime.robots.go2.agent.control import (
    ActionExecutor,
    ControlServerConfig,
    DryRunTransport,
    RobotState,
    build_parser,
    config_from_args,
    ensure_cyclonedds_library_dir,
    valid_api_token_for_bind,
)
from embodied_runtime.robots.go2.agent.control import cli as control_cli
from embodied_runtime.robots.go2.agent.control.http import make_request_handler


def request_json(
    port: int,
    method: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_http_auth_state_motion_update_and_emergency_stop() -> None:
    token = "t" * 32
    executor = ActionExecutor(DryRunTransport(), operator_motion_ready=True)
    handler = make_request_handler(executor, token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        status, payload = request_json(port, "GET", "/health")
        assert status == HTTPStatus.UNAUTHORIZED
        assert payload == {"error": "unauthorized"}

        status, payload = request_json(port, "GET", "/v1/state", token=token)
        assert status == HTTPStatus.OK
        assert payload["transport"] == "dry-run"
        assert payload["robot_state_fresh"] is True

        status, payload = request_json(
            port,
            "POST",
            "/v1/actions",
            token=token,
            payload={
                "action": "stream_move",
                "vx": 0.1,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "duration_s": 1.0,
            },
        )
        assert status == HTTPStatus.ACCEPTED
        action_id = payload["action_id"]

        status, payload = request_json(
            port,
            "POST",
            "/v1/actions",
            token=token,
            payload={
                "action": "update_move",
                "action_id": action_id,
                "vx": 0.15,
                "vy": 0.0,
                "yaw_rate": 0.1,
            },
        )
        assert status == HTTPStatus.OK
        assert payload["accepted"] is True

        status, payload = request_json(
            port,
            "POST",
            "/v1/stop",
            token=token,
            payload={},
        )
        assert status == HTTPStatus.OK
        assert payload["stopped"] is True
        assert payload["operator_motion_ready"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
        executor.close()


def test_http_rejects_invalid_json_and_unknown_path() -> None:
    executor = ActionExecutor(DryRunTransport(), operator_motion_ready=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_request_handler(executor, None),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        connection.request(
            "POST",
            "/v1/actions",
            body=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == HTTPStatus.BAD_REQUEST
        assert json.loads(response.read()) == {"error": "invalid JSON request"}
        connection.close()

        status, payload = request_json(port, "GET", "/missing")
        assert status == HTTPStatus.NOT_FOUND
        assert payload == {"error": "not found"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
        executor.close()


def test_external_bind_requires_long_whitespace_free_token() -> None:
    assert valid_api_token_for_bind("127.0.0.1", None)
    assert not valid_api_token_for_bind("0.0.0.0", None)
    assert not valid_api_token_for_bind("0.0.0.0", "x" * 31)
    assert valid_api_token_for_bind("0.0.0.0", "x" * 32)
    assert not valid_api_token_for_bind("0.0.0.0", "x" * 31 + " ")


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
        "embodied_runtime.robots.go2.agent.control",
        "--mode",
        "live",
    ]
    assert captured["environ"]["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(tmp_path.resolve())
