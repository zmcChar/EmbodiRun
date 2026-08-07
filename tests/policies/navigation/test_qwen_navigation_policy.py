from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from typing import Any

import pytest

from embodied_runtime.policies.navigation.qwen.errors import QwenNavigationValidationError
from embodied_runtime.policies.navigation.qwen.policy import QwenNavigationPolicy
from embodied_runtime.tasks.navigation import (
    EncodedRGBFrame,
    NavigationObservation,
    NavigationRequest,
    Waypoint,
    WaypointPlan,
)


def _response(plan: Mapping[str, Any] | str) -> dict[str, Any]:
    content = plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False)
    return {"choices": [{"message": {"content": content}}]}


def _valid_plan() -> dict[str, Any]:
    return {
        "frame": "base_link",
        "waypoints": [{"x_m": 0.8, "y_m": 0.1, "yaw_rad": 0.2}],
        "terminal": False,
        "confidence": 0.82,
        "valid_for_s": 1.5,
    }


class _FakeHttpTransport:
    def __init__(self, *responses: Mapping[str, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.thread_ids: list[int] = []
        self.closed = False

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.thread_ids.append(threading.get_ident())
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_s": timeout_s,
            }
        )
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _request() -> NavigationRequest:
    observation = NavigationObservation(
        episode_id="go2:test-episode",
        sequence=7,
        reset=True,
        rgb_frames=(
            EncodedRGBFrame(
                sequence=7,
                captured_at_s=10.0,
                data=b"\xff\xd8fake-jpeg",
                width=640,
                height=360,
            ),
        ),
        robot_state={"standing": True, "yaw_rad": 0.2},
    )
    return NavigationRequest("  导航到三脚架前  ", observation)


def test_policy_is_lazy_async_and_returns_shared_waypoint_contract() -> None:
    async def scenario() -> None:
        event_loop_thread = threading.get_ident()
        transport = _FakeHttpTransport(_response(_valid_plan()))
        factory_calls: list[bool] = []

        def transport_factory() -> _FakeHttpTransport:
            factory_calls.append(True)
            return transport

        policy = QwenNavigationPolicy(
            base_url="https://qwen.example.invalid/v1",
            model="qwen-nav-test",
            api_key="test-token",
            transport_factory=transport_factory,
        )
        assert not policy.connected
        assert factory_calls == []

        plan = await policy.plan(_request())

        assert plan == WaypointPlan(
            observation_sequence=7,
            waypoints=(Waypoint(x_m=0.8, y_m=0.1, yaw_rad=0.2),),
            terminal=False,
            confidence=0.82,
            valid_for_s=1.5,
        )
        assert policy.connected
        assert factory_calls == [True]
        assert len(transport.thread_ids) == 1
        assert transport.thread_ids[0] != event_loop_thread

        call = transport.calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "https://qwen.example.invalid/v1/chat/completions"
        assert call["headers"]["Authorization"] == "Bearer test-token"
        wire_request = call["payload"]
        assert wire_request["response_format"]["type"] == "json_schema"
        assert wire_request["response_format"]["json_schema"]["strict"] is True
        schema = wire_request["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["waypoints"]["items"]["additionalProperties"] is False

        user_content = wire_request["messages"][1]["content"]
        task = json.loads(user_content[0]["text"])
        assert task["instruction"] == "  导航到三脚架前  "
        assert task["observation"]["episode_id"] == "go2:test-episode"
        assert task["observation"]["sequence"] == 7
        assert task["observation"]["reset"] is True
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        await policy.aclose()
        assert transport.closed
        assert not policy.connected

    asyncio.run(scenario())


def test_policy_plan_exposes_the_task_semantic_interface() -> None:
    async def scenario() -> None:
        transport = _FakeHttpTransport(_response(_valid_plan()))
        policy = QwenNavigationPolicy(transport=transport)

        plan = await policy.plan(_request())

        assert isinstance(plan, WaypointPlan)
        assert plan.observation_sequence == 7
        assert plan.waypoints == (Waypoint(x_m=0.8, y_m=0.1, yaw_rad=0.2),)
        await policy.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid_content",
    [
        json.dumps({**_valid_plan(), "unexpected": "field"}),
        json.dumps({**_valid_plan(), "confidence": True}),
        json.dumps({**_valid_plan(), "waypoints": []}),
        ('{"frame":"base_link","waypoints":[],"terminal":true,"confidence":NaN,"valid_for_s":1.0}'),
    ],
)
def test_model_output_is_strictly_revalidated_after_schema_generation(
    invalid_content: str,
) -> None:
    async def scenario() -> None:
        transport = _FakeHttpTransport(
            _response(invalid_content),
            _response(invalid_content),
        )
        policy = QwenNavigationPolicy(transport=transport)

        with pytest.raises(QwenNavigationValidationError, match="two invalid"):
            await policy.plan(_request())

        assert len(transport.calls) == 2
        retry = transport.calls[1]["payload"]
        assert retry["messages"][-2]["role"] == "assistant"
        assert retry["messages"][-1]["role"] == "user"
        await policy.aclose()

    asyncio.run(scenario())


def test_close_before_first_inference_does_not_construct_transport() -> None:
    async def scenario() -> None:
        factory_calls: list[bool] = []

        def transport_factory() -> _FakeHttpTransport:
            factory_calls.append(True)
            return _FakeHttpTransport(_response(_valid_plan()))

        policy = QwenNavigationPolicy(transport_factory=transport_factory)
        await policy.aclose()
        await policy.aclose()

        assert factory_calls == []
        with pytest.raises(RuntimeError, match="closed"):
            await policy.plan(_request())

    asyncio.run(scenario())
