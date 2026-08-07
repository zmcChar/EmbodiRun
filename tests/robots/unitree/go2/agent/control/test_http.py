from __future__ import annotations

import http.client
import json
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any

from embodied_runtime.robots.unitree.go2.agent.control import (
    ActionExecutor,
    DryRunTransport,
    valid_api_token_for_bind,
)
from embodied_runtime.robots.unitree.go2.agent.control.http import make_request_handler


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
