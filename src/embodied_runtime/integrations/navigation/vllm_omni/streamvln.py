"""StreamVLN navigation policy backed by a vLLM-Omni OpenPI service."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from embodied_runtime.distributed.communication import OpenPiWebSocketEndpoint
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.models.vln.streamvln import MAX_FUTURE_ACTIONS
from embodied_runtime.policies.navigation.episode import EpisodeCursor
from embodied_runtime.policies.navigation.streamvln.geometry import (
    streamvln_actions_to_waypoint_plan,
)
from embodied_runtime.policies.navigation.validation import navigation_request
from embodied_runtime.policies.navigation.vision import decode_rgb
from embodied_runtime.tasks.navigation import NavigationRequest, WaypointPlan

from .codec import STREAMVLN_ACTION_HEAD, STREAMVLN_PADDING_ID, decode_streamvln_actions
from .common import (
    DEFAULT_VLLM_OMNI_URL,
    validate_image_payload,
    validate_navigation_handshake,
)


class VllmOmniStreamVLNNavigationPolicy:
    """Serialize one recurrent StreamVLN session over OpenPI."""

    def __init__(
        self,
        *,
        url: str = DEFAULT_VLLM_OMNI_URL,
        timeout_s: float = 120.0,
        session_id: str | None = None,
        endpoint: Any | None = None,
    ) -> None:
        if endpoint is not None and (
            url != DEFAULT_VLLM_OMNI_URL or timeout_s != 120.0 or session_id is not None
        ):
            raise ValueError("endpoint and OpenPI connection options are mutually exclusive")
        endpoint_session_id = getattr(endpoint, "session_id", None)
        self._session_id = endpoint_session_id or session_id or uuid.uuid4().hex
        self.endpoint = endpoint or OpenPiWebSocketEndpoint(
            url,
            timeout_s=timeout_s,
            session_id=self._session_id,
            # StreamVLN advances recurrent state. Retrying a request after an
            # ambiguous transport failure could advance twice or reconnect and
            # silently lose history, so requests are never replayed here.
            reconnect_attempts=0,
            expected_batch_size=1,
            expected_action_horizon=MAX_FUTURE_ACTIONS,
            expected_action_dim=1,
        )
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._cursor = EpisodeCursor()
        self._closed = False
        self._prepared = False
        self._remote_state_uncertain = False

    async def _prepare_locked(self) -> None:
        metadata = await self.endpoint.connect()
        try:
            validate_navigation_handshake(
                metadata,
                model_family="streamvln",
                input_schema="rgb_uint8_hwc",
                action_head=STREAMVLN_ACTION_HEAD,
                action_horizon=MAX_FUTURE_ACTIONS,
                action_dim=1,
                padding_id=STREAMVLN_PADDING_ID,
            )
        except Exception:
            await self.endpoint.aclose()
            raise
        self._prepared = True

    async def prepare(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("vLLM-Omni StreamVLN policy is closed")
            await self._prepare_locked()

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        navigation = navigation_request(request)
        observation = navigation.observation
        async with self._lock:
            if self._closed:
                raise RuntimeError("vLLM-Omni StreamVLN policy is closed")
            # connect() is idempotent and revalidates metadata after a prior
            # transport failure established a new WebSocket connection.
            await self._prepare_locked()
            should_reset = self._cursor.requires_reset(observation)
            if self._remote_state_uncertain and not observation.reset:
                raise RuntimeError(
                    "vLLM-Omni StreamVLN session state is uncertain after a failed request; "
                    "retry with observation.reset=true"
                )
            try:
                if should_reset:
                    await self.endpoint.reset(
                        {"episode_id": observation.episode_id},
                        session_id=self._session_id,
                    )
                    self._remote_state_uncertain = False
                rgb = await asyncio.to_thread(decode_rgb, observation.latest_rgb)
                validate_image_payload((rgb,))
                client_request_id = uuid.uuid4().hex
                result = await self.endpoint.infer_async(
                    InferenceRequest(
                        request_id=client_request_id,
                        payload={
                            "client_request_id": client_request_id,
                            "episode_id": observation.episode_id,
                            "observation_sequence": observation.sequence,
                            "rgb": rgb,
                            "prompt": navigation.instruction,
                        },
                        deadline_s=self._timeout_s,
                        metadata={"session_id": self._session_id},
                    )
                )
                actions = decode_streamvln_actions(result.output["actions"])
                plan = streamvln_actions_to_waypoint_plan(
                    actions,
                    observation_sequence=observation.sequence,
                    max_actions=MAX_FUTURE_ACTIONS,
                )
            except BaseException:
                # Cancellation, a lost response, and invalid output are all
                # ambiguous after a recurrent request may have reached the
                # server. Require an explicit episode reset before continuing.
                self._remote_state_uncertain = True
                raise
            self._cursor.commit(observation)
            return plan

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cursor.clear()
            self._remote_state_uncertain = False
            await self.endpoint.aclose()


__all__ = ["VllmOmniStreamVLNNavigationPolicy"]
