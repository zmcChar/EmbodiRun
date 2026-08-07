"""Stateful local InternVLA-N1 policy producing metric waypoint plans."""

from __future__ import annotations

import asyncio

from embodied_runtime.models.vla.internvla_n1 import (
    InternVLARuntime,
    decode_depth_payload,
)
from embodied_runtime.tasks.navigation import NavigationRequest, WaypointPlan

from .episode import EpisodeCursor
from .errors import NavigationPolicyError
from .internvla_mapping import internvla_prediction_to_waypoint_plan
from .validation import navigation_request
from .vision import decode_rgb


class InternVLANavigationPolicy:
    def __init__(self, runtime: InternVLARuntime | None = None, **runtime_options: object) -> None:
        if runtime is not None and runtime_options:
            raise ValueError("runtime and runtime_options are mutually exclusive")
        self.runtime = runtime or InternVLARuntime(**runtime_options)
        self._lock = asyncio.Lock()
        self._cursor = EpisodeCursor()
        self._closed = False

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        """Advance one recurrent episode and return its semantic navigation plan."""

        navigation = navigation_request(request)
        observation = navigation.observation
        async with self._lock:
            if self._closed:
                raise RuntimeError("InternVLA policy is closed")
            should_reset = self._cursor.requires_reset(observation)

            def infer() -> WaypointPlan:
                rgb = decode_rgb(observation.latest_rgb)
                depth_m = None
                if self.runtime.spec.depth_required:
                    if observation.depth is None:
                        raise NavigationPolicyError("InternVLA NavDP requires registered depth")
                    depth_m = decode_depth_payload(
                        observation.depth,
                        expected_shape=(int(rgb.shape[0]), int(rgb.shape[1])),
                    )
                if should_reset:
                    self.runtime.reset()
                native = self.runtime.predict(rgb, depth_m, navigation.instruction)
                return internvla_prediction_to_waypoint_plan(
                    native,
                    observation.sequence,
                )

            plan = await asyncio.to_thread(infer)
            self._cursor.commit(observation)
        return plan

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            self._cursor.clear()


__all__ = ["InternVLANavigationPolicy"]
