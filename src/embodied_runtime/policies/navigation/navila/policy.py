"""Episode-aware NaVILA policy producing task-owned waypoint plans."""

from __future__ import annotations

import asyncio
from pathlib import Path

from embodied_runtime.models.vla.navila import DEFAULT_MODEL, NaVILARuntime, sample_episode_frames
from embodied_runtime.tasks.navigation import EncodedRGBFrame, NavigationRequest, WaypointPlan

from ..episode import EpisodeCursor
from ..validation import navigation_request
from ..vision import decode_rgb
from .mapping import navila_prediction_to_waypoint_plan

MAX_EPISODE_FRAMES = 4096
MAX_EPISODE_IMAGE_BYTES = 512 * 1024 * 1024


class NaVILANavigationPolicy:
    """Own encoded episode history while keeping the model runtime task-independent."""

    def __init__(
        self,
        runtime: NaVILARuntime | None = None,
        *,
        navila_root: str | Path | None = None,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cuda:0",
        cuda_memory_fraction: float | None = None,
        max_new_tokens: int = 32,
        local_files_only: bool = True,
    ) -> None:
        if runtime is None:
            if navila_root is None:
                raise ValueError("navila_root is required when runtime is not supplied")
            runtime = NaVILARuntime(
                navila_root=navila_root,
                model_path=model_path,
                device=device,
                cuda_memory_fraction=cuda_memory_fraction,
                max_new_tokens=max_new_tokens,
                local_files_only=local_files_only,
            )
        self.runtime = runtime
        self._lock = asyncio.Lock()
        self._cursor = EpisodeCursor()
        self._frames: tuple[EncodedRGBFrame, ...] = ()
        self._closed = False

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
        total_bytes = sum(len(frame.data) for frame in frames)
        if total_bytes > MAX_EPISODE_IMAGE_BYTES:
            raise RuntimeError("NaVILA encoded episode history exceeds 512 MiB")
        return tuple(frames)

    async def prepare(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("NaVILA policy is closed")
            await asyncio.to_thread(self.runtime.load)

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        navigation = navigation_request(request)
        observation = navigation.observation
        async with self._lock:
            if self._closed:
                raise RuntimeError("NaVILA policy is closed")
            should_reset = self._cursor.requires_reset(observation)
            history = self._history(tuple(observation.rgb_frames), reset=should_reset)
            sampled = sample_episode_frames(history)
            rgb_frames = await asyncio.to_thread(
                lambda: tuple(decode_rgb(frame) for frame in sampled)
            )
            prediction = await asyncio.to_thread(
                self.runtime.predict,
                rgb_frames,
                navigation.instruction,
            )
            plan = navila_prediction_to_waypoint_plan(
                prediction,
                observation.sequence,
            )
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
            await asyncio.to_thread(self.runtime.close)


__all__ = [
    "MAX_EPISODE_FRAMES",
    "MAX_EPISODE_IMAGE_BYTES",
    "NaVILANavigationPolicy",
]
