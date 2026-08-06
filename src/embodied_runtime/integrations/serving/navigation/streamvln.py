"""Stateful local StreamVLN provider producing the shared waypoint contract."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    ModelSpec,
    Waypoint,
    WaypointPlan,
)
from embodied_runtime.integrations.serving.base import ProviderCapabilities
from embodied_runtime.models.vln.streamvln import DEFAULT_MODEL, StreamVLNRuntime

from .common import EpisodeCursor, decode_rgb, navigation_request


class StreamVLNNavigationProvider:
    """Serialize recurrent inference while keeping model code robot-independent."""

    def __init__(
        self,
        runtime: StreamVLNRuntime | None = None,
        *,
        streamvln_root: str | Path | None = None,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cuda:0",
        cuda_memory_fraction: float | None = None,
        max_new_tokens: int = 64,
        local_files_only: bool = False,
        warmup: bool = True,
    ) -> None:
        if runtime is None:
            if streamvln_root is None:
                raise ValueError("streamvln_root is required when runtime is not supplied")
            runtime = StreamVLNRuntime(
                streamvln_root=streamvln_root,
                model_path=model_path,
                device=device,
                cuda_memory_fraction=cuda_memory_fraction,
                max_new_tokens=max_new_tokens,
                local_files_only=local_files_only,
                warmup=warmup,
            )
        self.runtime = runtime
        self._lock = asyncio.Lock()
        self._cursor = EpisodeCursor()
        self._closed = False
        self._capabilities = ProviderCapabilities(
            name="streamvln-navigation",
            runtime="pytorch_streamvln",
            model=ModelSpec(
                model_id=str(self.runtime.config.model_path),
                family="vln",
                revision=self.runtime.config.revision,
                modalities=("vision", "language"),
                action_horizon=self.runtime.config.num_future_steps,
                metadata={"output_contract": "base_link_waypoint_plan"},
            ),
            is_remote=False,
            features=frozenset({"metric_waypoints", "recurrent_memory", "rgb"}),
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
                raise RuntimeError("StreamVLN provider is closed")
            should_reset = self._cursor.requires_reset(observation)

            def infer() -> WaypointPlan:
                if should_reset:
                    self.runtime.reset()
                prediction = self.runtime.predict(
                    decode_rgb(observation.latest_rgb), navigation.instruction
                )
                return WaypointPlan(
                    observation_sequence=observation.sequence,
                    waypoints=tuple(
                        Waypoint(point.x_m, point.y_m, point.yaw_rad)
                        for point in prediction.waypoints
                    ),
                    terminal=prediction.terminal,
                    confidence=1.0,
                    valid_for_s=5.0,
                )

            plan = await asyncio.to_thread(infer)
            self._cursor.commit(observation)
        return InferenceResult(
            request_id=request.request_id,
            output=plan,
            execution_time_s=time.monotonic() - started_at,
            metadata={
                "provider": "streamvln-navigation",
                "episode_id": observation.episode_id,
                "observation_sequence": observation.sequence,
            },
        )

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            self._cursor.clear()


__all__ = ["StreamVLNNavigationProvider"]
