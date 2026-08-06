from __future__ import annotations

import io
import json
from email.message import Message
from unittest.mock import patch

import pytest

from embodied_runtime.utils import HttpClientError, JsonHttpClient


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode())
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, amount: int) -> bytes:
        return self._body.read(amount)


def test_json_client_adds_auth_and_decodes_object() -> None:
    client = JsonHttpClient("http://robot:8080/", token="secret")
    with patch("urllib.request.urlopen", return_value=_Response({"ready": True})) as opened:
        assert client.request_json("GET", "/v1/state") == {"ready": True}
    request = opened.call_args.args[0]
    assert request.full_url == "http://robot:8080/v1/state"
    assert request.get_header("Authorization") == "Bearer secret"


def test_json_client_rejects_non_object_response() -> None:
    client = JsonHttpClient("http://robot:8080")
    with (
        patch("urllib.request.urlopen", return_value=_Response([1, 2])),
        pytest.raises(HttpClientError, match="JSON object"),
    ):
        client.request_json("GET", "/v1/state")
