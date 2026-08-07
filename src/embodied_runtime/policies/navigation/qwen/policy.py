"""Qwen navigation policy with a generic-inference compatibility method."""

from __future__ import annotations

import asyncio
import contextlib

from embodied_runtime.tasks.navigation import NavigationRequest, WaypointPlan

from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, QwenNavigationClient
from .inputs import (
    PreparedNavigationInput,
    prepare_navigation_request,
)
from .transport import JsonHttpTransport, TransportFactory


class QwenNavigationPolicy:
    """Generate validated metric waypoint plans without executing robot actions."""

    policy_name = "qwen-navigation"

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
        self._state = asyncio.Condition()
        self._close_lock = asyncio.Lock()
        self._active_requests = 0
        self._closed = False
        self._resources_closed = False

    @property
    def connected(self) -> bool:
        return not self._closed and self.client.connected

    async def prepare(self) -> None:
        """Validate lifecycle state without managing the external Qwen server."""

        async with self._state:
            if self._closed:
                raise RuntimeError(f"{self.policy_name} policy is closed")

    async def _admit(self) -> None:
        async with self._state:
            if self._closed:
                raise RuntimeError(f"{self.policy_name} policy is closed")
            self._active_requests += 1

    async def _release(self) -> None:
        async with self._state:
            self._active_requests -= 1
            self._state.notify_all()

    async def _execute_prepared(
        self,
        prepared: PreparedNavigationInput,
        *,
        task_name: str,
    ) -> WaypointPlan:
        inference = asyncio.create_task(
            asyncio.to_thread(
                self.client.plan,
                prepared.prompt,
                prepared.image_data,
                prepared.media_type,
                prepared.context,
                observation_sequence=prepared.observation_sequence,
            ),
            name=task_name,
        )
        try:
            return await asyncio.shield(inference)
        except asyncio.CancelledError:
            # Worker threads cannot be force-cancelled. Drain before closing
            # the underlying HTTP transport.
            with contextlib.suppress(Exception):
                await inference
            raise

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        """Return a semantic waypoint plan without a serving envelope."""

        if not isinstance(request, NavigationRequest):
            raise TypeError("request must be a NavigationRequest")
        await self._admit()
        try:
            prepared = prepare_navigation_request(request)
            return await self._execute_prepared(
                prepared,
                task_name=(
                    f"qwen-navigation-{prepared.episode_id}:{prepared.observation_sequence}"
                ),
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


__all__ = ["QwenNavigationPolicy"]
