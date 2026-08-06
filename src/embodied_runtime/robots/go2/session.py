"""Continuous language-conditioned navigation sessions for a Go2 robot.

The model and control loops deliberately have different cadences.  A slow
inference request may replace the follower's plan when it completes, while the
control loop continues to track the previously accepted world-frame plan.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    NavigationObservation,
    NavigationRequest,
    RequestStatus,
    WaypointPlan,
)
from embodied_runtime.integrations.serving.base import InferenceProvider
from embodied_runtime.utils import Pose2D

from .camera import Go2CameraClient
from .client import Go2ClientError, Go2ControlClient
from .motion import Go2VelocityLease
from .time_sync import capture_pose
from .types import Go2State
from .waypoint_follower import WorldWaypointFollower

NavigationEndReason = Literal["terminal", "max_runtime"]


class Go2NavigationSessionError(RuntimeError):
    """A camera, provider, or session invariant prevented navigation."""


@dataclass(frozen=True, slots=True)
class Go2NavigationSessionConfig:
    """Timing and execution policy for one navigation episode."""

    control_hz: float = 10.0
    max_runtime_s: float = 120.0
    execute: bool = False
    lease_duration_s: float = 10.0
    max_events: int = 256

    def __post_init__(self) -> None:
        for name in ("control_hz", "max_runtime_s", "lease_duration_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.execute, bool):
            raise TypeError("execute must be a boolean")
        if (
            isinstance(self.max_events, bool)
            or not isinstance(self.max_events, int)
            or self.max_events < 1
        ):
            raise ValueError("max_events must be a positive integer")


DEFAULT_GO2_NAVIGATION_SESSION_CONFIG = Go2NavigationSessionConfig()


@dataclass(frozen=True, slots=True)
class NavigationSessionEvent:
    """One bounded, CLI-friendly session event."""

    kind: str
    elapsed_s: float
    observation_sequence: int | None = None
    waypoint_index: int | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class Go2NavigationSessionResult:
    """Summary returned after a terminal action or runtime limit."""

    episode_id: str
    reason: NavigationEndReason
    elapsed_s: float
    inference_count: int
    plans_accepted: int
    control_ticks: int
    motion_commands: int
    last_observation_sequence: int | None
    events: tuple[NavigationSessionEvent, ...]
    events_dropped: int = 0


def _robot_state_metadata(state: Go2State) -> dict[str, object]:
    """Expose a stable state subset even when a fake/API omits raw metadata."""

    metadata = dict(state.raw)
    metadata.update(
        {
            "position": [state.pose.x_m, state.pose.y_m],
            "yaw": state.pose.yaw_rad,
            "velocity": [state.forward_velocity_mps, state.lateral_velocity_mps],
            "yaw_rate": state.yaw_rate_rps,
            "sequence": state.sequence,
        }
    )
    if state.received_at_s is not None:
        metadata["received_at_unix"] = state.received_at_s
    return metadata


class Go2NavigationSession:
    """Run one recurrent navigation episode with an independent control loop.

    The synchronous camera and robot clients are always dispatched to worker
    threads so local or remote model inference cannot starve the asyncio loop.
    ``execute=False`` is a strict dry-run: no lease method, including ``stop``,
    is called.
    """

    def __init__(
        self,
        provider: InferenceProvider,
        camera: Go2CameraClient,
        control: Go2ControlClient,
        *,
        follower: WorldWaypointFollower | None = None,
        config: Go2NavigationSessionConfig = DEFAULT_GO2_NAVIGATION_SESSION_CONFIG,
        event_sink: Callable[[NavigationSessionEvent], None] | None = None,
    ) -> None:
        if not isinstance(config, Go2NavigationSessionConfig):
            raise TypeError("config must be a Go2NavigationSessionConfig")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self.provider = provider
        self.camera = camera
        self.control = control
        self.follower = WorldWaypointFollower() if follower is None else follower
        self.config = config
        self.event_sink = event_sink
        self._lease = Go2VelocityLease(control, duration_s=config.lease_duration_s)
        self._running = False
        self._events: deque[NavigationSessionEvent] = deque(maxlen=config.max_events)
        self._events_dropped = 0
        self._started_at_s = 0.0
        self._inference_count = 0
        self._plans_accepted = 0
        self._control_ticks = 0
        self._motion_commands = 0
        self._last_observation_sequence: int | None = None
        self._reason: NavigationEndReason | None = None
        self._motion_authorized = False

    @property
    def events(self) -> tuple[NavigationSessionEvent, ...]:
        """The retained tail of events, including events from a failed run."""

        return tuple(self._events)

    def _emit(
        self,
        kind: str,
        *,
        observation_sequence: int | None = None,
        waypoint_index: int | None = None,
        message: str = "",
    ) -> None:
        if len(self._events) == self._events.maxlen:
            self._events_dropped += 1
        event = NavigationSessionEvent(
            kind=kind,
            elapsed_s=max(0.0, time.monotonic() - self._started_at_s),
            observation_sequence=observation_sequence,
            waypoint_index=waypoint_index,
            message=message,
        )
        self._events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)

    @staticmethod
    async def _wait_until_stopped(stop_event: asyncio.Event, delay_s: float) -> None:
        if delay_s <= 0:
            await asyncio.sleep(0)
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay_s)
        except asyncio.TimeoutError:
            pass

    async def _capture(
        self,
        episode_id: str,
        *,
        reset: bool,
    ) -> tuple[NavigationObservation, Pose2D]:
        before = await asyncio.to_thread(self.control.state)
        observation = await asyncio.to_thread(
            self.camera.capture,
            episode_id=episode_id,
            reset=reset,
            robot_state=_robot_state_metadata(before),
        )
        after = await asyncio.to_thread(self.control.state)
        if not isinstance(observation, NavigationObservation):
            raise Go2NavigationSessionError("camera must return a NavigationObservation")
        if observation.episode_id != episode_id:
            raise Go2NavigationSessionError("camera changed the navigation episode_id")
        if observation.reset is not reset:
            raise Go2NavigationSessionError("camera changed the requested reset flag")
        previous_sequence = self._last_observation_sequence
        if previous_sequence is not None and observation.sequence <= previous_sequence:
            raise Go2NavigationSessionError("camera observation sequence must strictly increase")
        self._last_observation_sequence = observation.sequence
        anchor = capture_pose(before, after, observation.latest_rgb.captured_at_s)
        self._emit("observation_captured", observation_sequence=observation.sequence)
        return observation, anchor

    async def _inference_loop(
        self,
        instruction: str,
        episode_id: str,
        stop_event: asyncio.Event,
    ) -> None:
        reset = True
        while not stop_event.is_set():
            observation, anchor = await self._capture(episode_id, reset=reset)
            reset = False
            request = InferenceRequest(
                payload=NavigationRequest(instruction, observation),
                request_id=f"{episode_id}:{observation.sequence}",
            )
            self._emit("inference_started", observation_sequence=observation.sequence)
            result = await self.provider.infer_async(request)
            self._inference_count += 1
            if not isinstance(result, InferenceResult):
                raise Go2NavigationSessionError("provider must return an InferenceResult")
            if result.status is not RequestStatus.SUCCEEDED:
                raise Go2NavigationSessionError(
                    f"provider returned non-success status {result.status!s}"
                )
            plan = result.output
            if not isinstance(plan, WaypointPlan):
                raise Go2NavigationSessionError("provider output must be a WaypointPlan")
            if plan.observation_sequence != observation.sequence:
                raise Go2NavigationSessionError(
                    "waypoint plan sequence must match its navigation observation"
                )
            # No await occurs between validation and replacement: from the
            # control task's perspective the complete plan changes atomically.
            self.follower.replace(plan, anchor, time.monotonic())
            self._plans_accepted += 1
            self._emit(
                "plan_accepted",
                observation_sequence=plan.observation_sequence,
                message=f"waypoints={len(plan.waypoints)} terminal={plan.terminal}",
            )
            if plan.terminal:
                # A terminal plan may still contain a final waypoint.  Stop
                # inference, but leave the fast loop tracking it to completion.
                await stop_event.wait()
                return

    async def _control_loop(self, stop_event: asyncio.Event) -> None:
        period_s = 1.0 / self.config.control_hz
        next_tick_s = time.monotonic()
        last_waypoint: tuple[int | None, int] | None = None
        while not stop_event.is_set():
            state = await asyncio.to_thread(self.control.state)
            sampled_at_s = time.monotonic()
            sample = self.follower.sample(state.pose, sampled_at_s)
            self._control_ticks += 1
            if self.config.execute:
                await asyncio.to_thread(self._lease.send, sample.command)
                self._motion_commands += 1
            progress = (self.follower.sequence, sample.waypoint_index)
            if progress != last_waypoint:
                self._emit(
                    "control_progress",
                    observation_sequence=self.follower.sequence,
                    waypoint_index=sample.waypoint_index,
                )
                last_waypoint = progress
            if sample.terminal_reached:
                self._reason = "terminal"
                self._emit(
                    "terminal_reached",
                    observation_sequence=self.follower.sequence,
                    waypoint_index=sample.waypoint_index,
                )
                stop_event.set()
                return
            next_tick_s += period_s
            now_s = time.monotonic()
            if next_tick_s < now_s - period_s:
                next_tick_s = now_s
            await self._wait_until_stopped(stop_event, next_tick_s - now_s)

    async def _runtime_limit(self, stop_event: asyncio.Event) -> None:
        await self._wait_until_stopped(stop_event, self.config.max_runtime_s)
        if not stop_event.is_set():
            self._reason = "max_runtime"
            self._emit("max_runtime")
            stop_event.set()
            # Local model inference may still be finishing in a worker thread.
            # Revoke physical motion immediately instead of waiting for that
            # slow inference call to return and the task group to drain.
            await self._force_stop()

    async def _force_stop(self) -> None:
        if not self._motion_authorized:
            return
        # Shield the physical stop attempt from cancellation of the owner task.
        await asyncio.shield(asyncio.to_thread(self._lease.stop))
        self._motion_authorized = False

    async def run(
        self,
        instruction: str,
        *,
        episode_id: str | None = None,
    ) -> Go2NavigationSessionResult:
        """Run exactly one episode; the same instance may be reused sequentially."""

        if self._running:
            raise RuntimeError("this navigation session is already running")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if episode_id is None:
            episode_id = f"go2-{uuid.uuid4().hex}"
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")

        self._running = True
        self._events.clear()
        self._events_dropped = 0
        self._started_at_s = time.monotonic()
        self._inference_count = 0
        self._plans_accepted = 0
        self._control_ticks = 0
        self._motion_commands = 0
        self._last_observation_sequence = None
        self._reason = None
        self._motion_authorized = False
        self.follower.reset_episode()
        stop_event = asyncio.Event()

        self._emit("session_started", message=f"execute={self.config.execute}")
        if self.config.execute:
            try:
                await asyncio.to_thread(self.control.preflight)
            except BaseException as error:
                # Preflight is read-only.  It happens before a lease exists, so
                # an interlock failure must not itself cause a motion POST.
                self._emit("preflight_failed", message=repr(error))
                self._running = False
                raise
            self._motion_authorized = True
            self._emit("preflight_passed")

        tasks = (
            asyncio.create_task(
                self._inference_loop(instruction.strip(), episode_id, stop_event),
                name="go2-navigation-inference",
            ),
            asyncio.create_task(self._control_loop(stop_event), name="go2-navigation-control"),
            asyncio.create_task(self._runtime_limit(stop_event), name="go2-navigation-timeout"),
        )

        try:
            await asyncio.gather(*tasks)
        except BaseException as error:
            stop_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self._force_stop()
            except Go2ClientError as stop_error:
                error.add_note(f"Go2 forced stop also failed: {stop_error!r}")
            self._emit("session_failed", message=repr(error))
            raise
        else:
            await self._force_stop()
        finally:
            self._running = False

        if self._reason is None:
            raise Go2NavigationSessionError("navigation ended without a terminal reason")
        self._emit("session_stopped", message=self._reason)
        return Go2NavigationSessionResult(
            episode_id=episode_id,
            reason=self._reason,
            elapsed_s=time.monotonic() - self._started_at_s,
            inference_count=self._inference_count,
            plans_accepted=self._plans_accepted,
            control_ticks=self._control_ticks,
            motion_commands=self._motion_commands,
            last_observation_sequence=self._last_observation_sequence,
            events=tuple(self._events),
            events_dropped=self._events_dropped,
        )


__all__ = [
    "DEFAULT_GO2_NAVIGATION_SESSION_CONFIG",
    "Go2NavigationSession",
    "Go2NavigationSessionConfig",
    "Go2NavigationSessionError",
    "Go2NavigationSessionResult",
    "NavigationEndReason",
    "NavigationSessionEvent",
]
