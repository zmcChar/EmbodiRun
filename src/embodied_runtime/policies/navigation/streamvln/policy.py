"""Stateful local StreamVLN policy producing the shared waypoint contract."""

from __future__ import annotations

import asyncio
from pathlib import Path

from embodied_runtime.models.vln.streamvln import DEFAULT_MODEL, StreamVLNRuntime
from embodied_runtime.tasks.navigation import NavigationRequest, WaypointPlan

from ..episode import EpisodeCursor
from ..validation import navigation_request
from ..vision import decode_rgb
from .geometry import streamvln_actions_to_waypoint_plan


class StreamVLNNavigationPolicy:
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

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        """Advance one recurrent episode and return its semantic navigation plan."""

        navigation = navigation_request(request)
        observation = navigation.observation
        async with self._lock:
            if self._closed:
                raise RuntimeError("StreamVLN policy is closed")
            should_reset = self._cursor.requires_reset(observation)

            def infer() -> WaypointPlan:
                if should_reset:
                    self.runtime.reset()
                prediction = self.runtime.predict(
                    decode_rgb(observation.latest_rgb), navigation.instruction
                )
                return streamvln_actions_to_waypoint_plan(
                    prediction.actions,
                    observation_sequence=observation.sequence,
                    max_actions=self.runtime.config.num_future_steps,
                )

            plan = await asyncio.to_thread(infer)
            self._cursor.commit(observation)
        return plan

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            self._cursor.clear()


__all__ = ["StreamVLNNavigationPolicy"]
