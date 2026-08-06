"""Stateful local InternVLA-N1 provider producing metric waypoint plans."""

from __future__ import annotations

import asyncio
import time

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    ModelSpec,
    Waypoint,
    WaypointPlan,
)
from embodied_runtime.integrations.serving.base import ProviderCapabilities
from embodied_runtime.models.vla.internvla_n1 import (
    InternVLARuntime,
    convert_native_prediction,
    decode_depth_payload,
)

from .common import EpisodeCursor, NavigationProviderError, decode_rgb, navigation_request


class InternVLANavigationProvider:
    def __init__(self, runtime: InternVLARuntime | None = None, **runtime_options: object) -> None:
        if runtime is not None and runtime_options:
            raise ValueError("runtime and runtime_options are mutually exclusive")
        self.runtime = runtime or InternVLARuntime(**runtime_options)
        self._lock = asyncio.Lock()
        self._cursor = EpisodeCursor()
        self._closed = False
        spec = self.runtime.spec
        features = {"metric_waypoints", "recurrent_memory", "rgb"}
        if spec.depth_required:
            features.add("registered_depth")
        self._capabilities = ProviderCapabilities(
            name="internvla-navigation",
            runtime="pytorch_internvla_n1",
            model=ModelSpec(
                model_id=spec.model_id,
                family="vla_vln",
                modalities=("vision", "depth", "language")
                if spec.depth_required
                else ("vision", "language"),
                metadata={
                    "variant": spec.name,
                    "system1": spec.system1,
                    "output_contract": "base_link_waypoint_plan",
                },
            ),
            is_remote=False,
            features=frozenset(features),
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        navigation = navigation_request(request.payload)
        observation = navigation.observation
        started_at = time.monotonic()
        async with self._lock:
            if self._closed:
                raise RuntimeError("InternVLA provider is closed")
            should_reset = self._cursor.requires_reset(observation)

            def infer() -> WaypointPlan:
                rgb = decode_rgb(observation.latest_rgb)
                depth_m = None
                if self.runtime.spec.depth_required:
                    if observation.depth is None:
                        raise NavigationProviderError("InternVLA NavDP requires registered depth")
                    depth_m = decode_depth_payload(
                        observation.depth,
                        expected_shape=(int(rgb.shape[0]), int(rgb.shape[1])),
                    )
                if should_reset:
                    self.runtime.reset()
                native = self.runtime.predict(rgb, depth_m, navigation.instruction)
                converted = convert_native_prediction(native, observation.sequence)
                return WaypointPlan(
                    observation_sequence=converted.observation_sequence,
                    waypoints=tuple(
                        Waypoint(point.x_m, point.y_m, point.yaw_rad)
                        for point in converted.waypoints
                    ),
                    terminal=converted.terminal,
                    confidence=converted.confidence,
                    valid_for_s=converted.valid_for_s,
                )

            plan = await asyncio.to_thread(infer)
            self._cursor.commit(observation)
        return InferenceResult(
            request_id=request.request_id,
            output=plan,
            execution_time_s=time.monotonic() - started_at,
            metadata={
                "provider": "internvla-navigation",
                "variant": self.runtime.variant,
                "episode_id": observation.episode_id,
                "observation_sequence": observation.sequence,
            },
        )

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            self._cursor.clear()


__all__ = ["InternVLANavigationProvider"]
