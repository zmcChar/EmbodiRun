"""Qwen navigation provider over an OpenAI-compatible HTTP endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    ModelSpec,
    RawRequest,
)
from embodied_runtime.contracts.navigation import (
    EncodedDepthFrame,
    EncodedRGBFrame,
    NavigationObservation,
    NavigationRequest,
)
from embodied_runtime.integrations.serving.base import ProviderCapabilities

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    JsonHttpTransport,
    QwenNavigationClient,
    TransportFactory,
)
from .schema import QwenNavigationError


@dataclass(frozen=True, slots=True)
class _NavigationInput:
    prompt: str
    image_data: bytes
    media_type: str
    observation_sequence: int
    episode_id: str
    context: Mapping[str, Any]


def _depth_context(depth: EncodedDepthFrame | None) -> dict[str, Any] | None:
    if depth is None:
        return None
    return {
        "sequence": depth.sequence,
        "captured_at_s": depth.captured_at_s,
        "width": depth.width,
        "height": depth.height,
        "scale_m": depth.scale_m,
        "registered_to_rgb": depth.registered_to_rgb,
        "encoding": depth.encoding,
        "metadata": dict(depth.metadata),
    }


def _navigation_observation_input(
    prompt: str,
    observation: NavigationObservation,
) -> _NavigationInput:
    latest = observation.latest_rgb
    context = {
        "episode_id": observation.episode_id,
        "sequence": observation.sequence,
        "reset": observation.reset,
        "rgb": {
            "sequence": latest.sequence,
            "captured_at_s": latest.captured_at_s,
            "media_type": latest.media_type,
            "width": latest.width,
            "height": latest.height,
            "context_frame_count": len(observation.rgb_frames),
        },
        "depth": _depth_context(observation.depth),
        "robot_state": dict(observation.robot_state),
        "metadata": dict(observation.metadata),
    }
    return _NavigationInput(
        prompt=prompt,
        image_data=latest.data,
        media_type=latest.media_type,
        observation_sequence=observation.sequence,
        episode_id=observation.episode_id,
        context=context,
    )


def _nonnegative_sequence(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QwenNavigationError("observation sequence must be a non-negative integer")
    return value


def _raw_rgb(
    observation: Mapping[str, Any],
) -> tuple[bytes, str, dict[str, Any], int | None]:
    raw_rgb: object | None = None
    for name in ("latest_rgb", "rgb", "image", "image_bytes"):
        if name in observation:
            raw_rgb = observation[name]
            break
    if raw_rgb is None:
        raise QwenNavigationError(
            "RawRequest observation requires latest_rgb, rgb, image, or image_bytes"
        )

    if isinstance(raw_rgb, EncodedRGBFrame):
        return (
            raw_rgb.data,
            raw_rgb.media_type,
            {
                "sequence": raw_rgb.sequence,
                "captured_at_s": raw_rgb.captured_at_s,
                "media_type": raw_rgb.media_type,
                "width": raw_rgb.width,
                "height": raw_rgb.height,
            },
            raw_rgb.sequence,
        )

    if isinstance(raw_rgb, Mapping):
        image_data = raw_rgb.get("data")
        media_type = raw_rgb.get("media_type", observation.get("media_type", "image/jpeg"))
        sequence_value = raw_rgb.get("sequence")
        rgb_context = {key: value for key, value in raw_rgb.items() if key not in {"data", "bytes"}}
        if image_data is None:
            image_data = raw_rgb.get("bytes")
    else:
        image_data = raw_rgb
        media_type = observation.get("media_type", "image/jpeg")
        sequence_value = None
        rgb_context = {"media_type": media_type}

    if not isinstance(image_data, bytes):
        raise QwenNavigationError("RawRequest latest RGB data must be bytes")
    if not isinstance(media_type, str):
        raise QwenNavigationError("RawRequest RGB media_type must be a string")
    if sequence_value is not None:
        sequence_value = _nonnegative_sequence(sequence_value)
    rgb_context.setdefault("media_type", media_type)
    return image_data, media_type, rgb_context, sequence_value


def _raw_depth_context(value: object) -> object:
    if isinstance(value, EncodedDepthFrame):
        return _depth_context(value)
    return value


def _raw_request_input(raw: RawRequest) -> _NavigationInput:
    if not isinstance(raw.prompt, str) or not raw.prompt.strip():
        raise QwenNavigationError("RawRequest.prompt must be a non-empty string")

    observation = raw.observation
    if isinstance(observation, NavigationObservation):
        return _navigation_observation_input(raw.prompt, observation)
    if not isinstance(observation, Mapping):
        raise QwenNavigationError("RawRequest.observation must be a mapping")

    navigation_observation = observation.get("navigation_observation")
    if navigation_observation is not None:
        if not isinstance(navigation_observation, NavigationObservation):
            raise QwenNavigationError("navigation_observation must be a NavigationObservation")
        return _navigation_observation_input(raw.prompt, navigation_observation)

    image_data, media_type, rgb_context, frame_sequence = _raw_rgb(observation)
    sequence_value = observation.get(
        "observation_sequence",
        observation.get("sequence", frame_sequence if frame_sequence is not None else 0),
    )
    sequence = _nonnegative_sequence(sequence_value)
    if frame_sequence is not None and frame_sequence != sequence:
        raise QwenNavigationError("RawRequest latest RGB sequence must match observation sequence")
    episode_id = observation.get("episode_id", raw.metadata.get("episode_id", "default"))
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise QwenNavigationError("RawRequest episode_id must be a non-empty string")
    reset = observation.get("reset", False)
    if type(reset) is not bool:
        raise QwenNavigationError("RawRequest reset must be a boolean")

    raw_depth = observation.get("depth")
    if isinstance(raw_depth, EncodedDepthFrame) and raw_depth.sequence != sequence:
        raise QwenNavigationError("RawRequest depth sequence must match observation sequence")
    context = {
        "episode_id": episode_id,
        "sequence": sequence,
        "reset": reset,
        "rgb": rgb_context,
        "depth": _raw_depth_context(raw_depth),
        "robot_state": observation.get("robot_state", {}),
        "history": observation.get("history", []),
        "metadata": observation.get("metadata", dict(raw.metadata)),
    }
    return _NavigationInput(
        prompt=raw.prompt,
        image_data=image_data,
        media_type=media_type,
        observation_sequence=sequence,
        episode_id=episode_id,
        context=context,
    )


def _normalize_request(request: InferenceRequest) -> _NavigationInput:
    payload = request.payload
    if isinstance(payload, NavigationRequest):
        return _navigation_observation_input(payload.instruction, payload.observation)
    if isinstance(payload, RawRequest):
        return _raw_request_input(payload)
    raise TypeError("Qwen navigation provider payload must be NavigationRequest or RawRequest")


class QwenNavigationProvider:
    """Generate validated metric waypoint plans without executing robot actions."""

    provider_name = "qwen-navigation"
    provider_runtime = "openai_compatible_qwen"

    def __init__(
        self,
        client: QwenNavigationClient | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: JsonHttpTransport | None = None,
        transport_factory: TransportFactory | None = None,
    ) -> None:
        if client is not None and (transport is not None or transport_factory is not None):
            raise ValueError("an explicit client cannot be combined with transport configuration")
        self.client = client or QwenNavigationClient(
            base_url,
            model,
            api_key=api_key,
            timeout_s=timeout_s,
            transport=transport,
            transport_factory=transport_factory,
        )
        self._capabilities = ProviderCapabilities(
            name=type(self).provider_name,
            runtime=type(self).provider_runtime,
            model=ModelSpec(
                model_id=self.client.model,
                family="vln",
                modalities=("vision", "language"),
                metadata={
                    "output_contract": "base_link_waypoint_plan",
                    "structured_output": "json_schema",
                },
            ),
            is_remote=True,
            transport="openai_compatible_http",
            features=frozenset(
                {
                    "async_infer",
                    "lazy_connection",
                    "metric_waypoints",
                    "structured_output",
                }
            ),
        )
        self._state = asyncio.Condition()
        self._close_lock = asyncio.Lock()
        self._active_requests = 0
        self._closed = False
        self._resources_closed = False

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def connected(self) -> bool:
        return not self._closed and self.client.connected

    async def _admit(self) -> None:
        async with self._state:
            if self._closed:
                raise RuntimeError(f"{self.provider_name} provider is closed")
            self._active_requests += 1

    async def _release(self) -> None:
        async with self._state:
            self._active_requests -= 1
            self._state.notify_all()

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        await self._admit()
        try:
            prepared = _normalize_request(request)
            started_at = time.monotonic()
            inference = asyncio.create_task(
                asyncio.to_thread(
                    self.client.plan,
                    prepared.prompt,
                    prepared.image_data,
                    prepared.media_type,
                    prepared.context,
                    observation_sequence=prepared.observation_sequence,
                ),
                name=f"qwen-navigation-{request.request_id}",
            )
            try:
                plan = await asyncio.shield(inference)
            except asyncio.CancelledError:
                # A worker thread cannot be force-cancelled.  Drain it before
                # allowing aclose() to release the underlying HTTP transport.
                with contextlib.suppress(Exception):
                    await inference
                raise
            execution_time_s = time.monotonic() - started_at
            return InferenceResult(
                request_id=request.request_id,
                output=plan,
                execution_time_s=execution_time_s,
                metadata={
                    "provider": self.provider_name,
                    "provider_runtime": self.provider_runtime,
                    "model_id": self.client.model,
                    "episode_id": prepared.episode_id,
                    "observation_sequence": prepared.observation_sequence,
                    "output_contract": "base_link_waypoint_plan",
                },
            )
        finally:
            await self._release()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._resources_closed:
                return
            async with self._state:
                self._closed = True
                while self._active_requests:
                    await self._state.wait()
            await asyncio.to_thread(self.client.close)
            self._resources_closed = True


__all__ = ["QwenNavigationProvider"]
