from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from rlinf_deploy.inference import (
    ImagePayload,
    PolicyObservation,
    VvlaWirelessClient,
    VvlaWirelessError,
)


class FakeWirelessTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any] | None, float]] = []
        self.failure: VvlaWirelessError | None = None

    def request(self, method, payload, *, timeout_s):
        self.calls.append((method, payload, timeout_s))
        if self.failure is not None:
            raise self.failure
        if method == "open_session":
            return {"session_id": "session-1", "session_revision": 0}
        if method == "step":
            return {
                "request_id": payload["request_id"],
                "session_id": payload["session_id"],
                "step_id": payload["step_id"],
                "session_revision": 1,
                "action_space": "pi05.action_chunk.v1",
                "actions": [{"type": "action_chunk", "values": {"data": [[0.0]]}}],
                "timing": {"policy_ms": 1.0},
            }
        if method == "reset":
            return {"session_id": payload["session_id"], "session_revision": 2}
        if method == "close":
            return {"ok": True}
        return {"status": "ok"}

    def shutdown(self) -> None:
        pass


def test_wireless_client_sends_structured_image_segments() -> None:
    transport = FakeWirelessTransport()
    client = VvlaWirelessClient(transport, timeout_s=2.5)
    session = client.open_session(robot_id="fr3", action_space="pi05.action_chunk.v1")
    result = client.step(
        PolicyObservation(
            session_id=session.session_id,
            request_id="request-1",
            step_id=0,
            instruction="move",
            state={"joints": [0.0]},
            images=(ImagePayload("wrist", "image/jpeg", b"jpeg-bytes"),),
        )
    )

    method, payload, timeout_s = transport.calls[1]
    assert method == "step"
    assert payload["schema"] == "vvla.policy.step.v1"
    assert payload["images"] == [
        {"name": "wrist", "mime_type": "image/jpeg", "data": b"jpeg-bytes"}
    ]
    assert timeout_s == 2.5
    assert result.request_id == "request-1"


def test_wireless_client_preserves_remote_error_details() -> None:
    transport = FakeWirelessTransport()
    transport.failure = VvlaWirelessError(
        "step_id must be monotonic",
        status=409,
        code="out_of_order_step",
    )
    client = VvlaWirelessClient(transport)

    with pytest.raises(VvlaWirelessError) as caught:
        client.capabilities()

    assert caught.value.status == 409
    assert caught.value.code == "out_of_order_step"
