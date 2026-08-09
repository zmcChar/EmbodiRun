"""NaVILA navigation policy backed by a vLLM-Omni OpenPI service."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from embodied_runtime.distributed.communication import OpenPiWebSocketEndpoint
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.models.vla.navila import NaVILAPrediction, sample_episode_frames
from embodied_runtime.policies.navigation.episode import EpisodeCursor
from embodied_runtime.policies.navigation.navila.mapping import (
    navila_prediction_to_waypoint_plan,
)
from embodied_runtime.policies.navigation.navila.policy import (
    MAX_EPISODE_FRAMES,
    MAX_EPISODE_IMAGE_BYTES,
)
from embodied_runtime.policies.navigation.validation import navigation_request
from embodied_runtime.policies.navigation.vision import decode_rgb
from embodied_runtime.tasks.navigation import EncodedRGBFrame, NavigationRequest, WaypointPlan

from .codec import NAVILA_ACTION_HEAD, decode_navila_action
from .common import (
    DEFAULT_VLLM_OMNI_URL,
    validate_image_payload,
    validate_navigation_handshake,
)


class VllmOmniNaVILANavigationPolicy:
    """Own NaVILA frame history while inference is served remotely."""

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
            reconnect_attempts=0,
            expected_batch_size=1,
            expected_action_horizon=1,
            expected_action_dim=3,
        )
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._cursor = EpisodeCursor()
        self._frames: tuple[EncodedRGBFrame, ...] = ()
        self._closed = False
        self._prepared = False

    @property
    def history_size(self) -> int:
        return len(self._frames)

    def _history(
        self,
        incoming: tuple[EncodedRGBFrame, ...],
        *,
        reset: bool,
    ) -> tuple[EncodedRGBFrame, ...]:
        frames = [] if reset else list(self._frames)
        known = {frame.sequence for frame in frames}
        for frame in incoming:
            if frame.sequence not in known:
                frames.append(frame)
                known.add(frame.sequence)
        if len(frames) > MAX_EPISODE_FRAMES:
            raise RuntimeError(f"NaVILA episode exceeds {MAX_EPISODE_FRAMES} frames")
        if sum(len(frame.data) for frame in frames) > MAX_EPISODE_IMAGE_BYTES:
            raise RuntimeError("NaVILA encoded episode history exceeds 512 MiB")
        return tuple(frames)

    async def _prepare_locked(self) -> None:
        metadata = await self.endpoint.connect()
        try:
            validate_navigation_handshake(
                metadata,
                model_family="navila",
                input_schema="rgb_frames_uint8_list_hwc",
                action_head=NAVILA_ACTION_HEAD,
                action_horizon=1,
                action_dim=3,
            )
        except Exception:
            await self.endpoint.aclose()
            raise
        self._prepared = True

    async def prepare(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("vLLM-Omni NaVILA policy is closed")
            await self._prepare_locked()

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        navigation = navigation_request(request)
        observation = navigation.observation
        async with self._lock:
            if self._closed:
                raise RuntimeError("vLLM-Omni NaVILA policy is closed")
            # connect() is idempotent and revalidates metadata after a prior
            # transport failure established a new WebSocket connection.
            await self._prepare_locked()
            should_reset = self._cursor.requires_reset(observation)
            if should_reset:
                await self.endpoint.reset(
                    {"episode_id": observation.episode_id},
                    session_id=self._session_id,
                )
            history = self._history(tuple(observation.rgb_frames), reset=should_reset)
            sampled = sample_episode_frames(history)

            def decode_frames() -> Any:
                return tuple(decode_rgb(frame) for frame in sampled)

            frames = await asyncio.to_thread(decode_frames)
            validate_image_payload(frames)
            client_request_id = uuid.uuid4().hex
            result = await self.endpoint.infer_async(
                InferenceRequest(
                    request_id=client_request_id,
                    payload={
                        "client_request_id": client_request_id,
                        "episode_id": observation.episode_id,
                        "observation_sequence": observation.sequence,
                        "rgb_frames": frames,
                        "prompt": navigation.instruction,
                    },
                    deadline_s=self._timeout_s,
                    metadata={"session_id": self._session_id},
                )
            )
            prediction = NaVILAPrediction(
                action=decode_navila_action(result.output["actions"]),
                raw_output="",
                token_ids=(),
                generation_time_s=result.execution_time_s,
            )
            plan = navila_prediction_to_waypoint_plan(prediction, observation.sequence)
            self._frames = history
            self._cursor.commit(observation)
            return plan

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._frames = ()
            self._cursor.clear()
            await self.endpoint.aclose()


__all__ = ["VllmOmniNaVILANavigationPolicy"]
