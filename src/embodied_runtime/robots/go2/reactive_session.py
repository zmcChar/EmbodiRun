"""Image-reactive Go2 navigation for robots without usable odometry.

Each cycle captures a fresh image, asks the navigation provider for a plan,
and executes only its first base-frame waypoint as a bounded velocity pulse.
The control service stops that pulse when its duration expires; the next image
is not captured until the pulse has settled.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    NavigationObservation,
    NavigationRequest,
    RequestStatus,
    Waypoint,
    WaypointPlan,
)
from embodied_runtime.integrations.serving.base import InferenceProvider

from .camera import Go2CameraClient
from .client import Go2ClientError, Go2ControlClient
from .session import (
    Go2NavigationSessionError,
    Go2NavigationSessionResult,
    NavigationEndReason,
    NavigationSessionEvent,
    _robot_state_metadata,
)
from .types import DEFAULT_GO2_LIMITS, BaseVelocityCommand, Go2Limits


@dataclass(frozen=True, slots=True)
class Go2ReactiveSessionConfig:
    """Timing and pulse policy for image-reactive navigation."""

    max_runtime_s: float = 120.0
    execute: bool = False
    linear_speed_mps: float = 0.30
    yaw_rate_rps: float = 0.60
    min_pulse_s: float = 0.05
    max_pulse_s: float = 1.50
    settle_s: float = 0.15
    max_events: int = 256
    limits: Go2Limits = DEFAULT_GO2_LIMITS

    def __post_init__(self) -> None:
        for name in (
            "max_runtime_s",
            "linear_speed_mps",
            "yaw_rate_rps",
            "min_pulse_s",
            "max_pulse_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if isinstance(self.settle_s, bool) or not isinstance(self.settle_s, (int, float)):
            raise TypeError("settle_s must be a number")
        settle_s = float(self.settle_s)
        if not math.isfinite(settle_s) or settle_s < 0:
            raise ValueError("settle_s must be finite and non-negative")
        object.__setattr__(self, "settle_s", settle_s)
        if self.min_pulse_s > self.max_pulse_s:
            raise ValueError("min_pulse_s must not exceed max_pulse_s")
        if self.linear_speed_mps > min(self.limits.max_abs_vx_mps, self.limits.max_abs_vy_mps):
            raise ValueError("linear_speed_mps exceeds Go2 limits")
        if self.yaw_rate_rps > self.limits.max_abs_yaw_rate_rps:
            raise ValueError("yaw_rate_rps exceeds Go2 limits")
        if not isinstance(self.execute, bool):
            raise TypeError("execute must be a boolean")
        if (
            isinstance(self.max_events, bool)
            or not isinstance(self.max_events, int)
            or self.max_events < 1
        ):
            raise ValueError("max_events must be a positive integer")


DEFAULT_GO2_REACTIVE_SESSION_CONFIG = Go2ReactiveSessionConfig()


def waypoint_to_velocity_pulse(
    waypoint: Waypoint,
    config: Go2ReactiveSessionConfig = DEFAULT_GO2_REACTIVE_SESSION_CONFIG,
) -> tuple[BaseVelocityCommand, float]:
    """Convert one capture-frame waypoint to one bounded open-loop pulse."""

    if not isinstance(waypoint, Waypoint):
        raise TypeError("waypoint must be a Waypoint")
    distance_m = math.hypot(waypoint.x_m, waypoint.y_m)
    required_s = max(
        distance_m / config.linear_speed_mps,
        abs(waypoint.yaw_rad) / config.yaw_rate_rps,
        config.min_pulse_s,
    )
    duration_s = min(required_s, config.max_pulse_s)

    def bounded(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    limits = config.limits
    return (
        BaseVelocityCommand(
            bounded(waypoint.x_m / duration_s, limits.max_abs_vx_mps),
            bounded(waypoint.y_m / duration_s, limits.max_abs_vy_mps),
            bounded(waypoint.yaw_rad / duration_s, limits.max_abs_yaw_rate_rps),
            limits,
        ),
        duration_s,
    )


class Go2ReactiveNavigationSession:
    """Replan after every short pulse instead of integrating Go2 odometry."""

    def __init__(
        self,
        provider: InferenceProvider,
        camera: Go2CameraClient,
        control: Go2ControlClient,
        *,
        config: Go2ReactiveSessionConfig = DEFAULT_GO2_REACTIVE_SESSION_CONFIG,
        event_sink: Callable[[NavigationSessionEvent], None] | None = None,
    ) -> None:
        self.provider = provider
        self.camera = camera
        self.control = control
        self.config = config
        self.event_sink = event_sink
        self._events: deque[NavigationSessionEvent] = deque(maxlen=config.max_events)
        self._events_dropped = 0
        self._started_at_s = 0.0
        self._last_sequence: int | None = None
        self._inference_count = 0
        self._plans_accepted = 0
        self._motion_commands = 0
        self._running = False
        self._motion_authorized = False

    @property
    def events(self) -> tuple[NavigationSessionEvent, ...]:
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

    async def _capture(self, episode_id: str, *, reset: bool) -> NavigationObservation:
        state = await asyncio.to_thread(self.control.state)
        observation = await asyncio.to_thread(
            self.camera.capture,
            episode_id=episode_id,
            reset=reset,
            robot_state=_robot_state_metadata(state),
        )
        if not isinstance(observation, NavigationObservation):
            raise Go2NavigationSessionError("camera must return a NavigationObservation")
        if observation.episode_id != episode_id or observation.reset is not reset:
            raise Go2NavigationSessionError("camera changed episode identity or reset flag")
        if self._last_sequence is not None and observation.sequence <= self._last_sequence:
            raise Go2NavigationSessionError("camera observation sequence must strictly increase")
        self._last_sequence = observation.sequence
        self._emit("observation_captured", observation_sequence=observation.sequence)
        return observation

    async def _infer(self, instruction: str, observation: NavigationObservation) -> WaypointPlan:
        request = InferenceRequest(
            payload=NavigationRequest(instruction, observation),
            request_id=f"{observation.episode_id}:{observation.sequence}",
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
        self._plans_accepted += 1
        self._emit(
            "plan_accepted",
            observation_sequence=observation.sequence,
            message=f"waypoints={len(plan.waypoints)} terminal={plan.terminal}",
        )
        return plan

    async def _cycle(self, instruction: str, episode_id: str) -> NavigationEndReason:
        reset = True
        while True:
            observation = await self._capture(episode_id, reset=reset)
            reset = False
            plan = await self._infer(instruction, observation)
            if not plan.waypoints:
                if not plan.terminal:  # Defensive: WaypointPlan already rejects this.
                    raise Go2NavigationSessionError("empty non-terminal waypoint plan")
                self._emit("terminal_reached", observation_sequence=observation.sequence)
                return "terminal"

            command, duration_s = waypoint_to_velocity_pulse(plan.waypoints[0], self.config)
            self._emit(
                "pulse_planned",
                observation_sequence=observation.sequence,
                waypoint_index=0,
                message=(
                    f"vx={command.vx_mps:.3f} vy={command.vy_mps:.3f} "
                    f"yaw_rate={command.yaw_rate_rps:.3f} duration_s={duration_s:.3f}"
                ),
            )
            if self.config.execute and command.moving:
                await asyncio.to_thread(
                    self.control.start_velocity_lease,
                    command,
                    duration_s=duration_s,
                )
                self._motion_commands += 1
                self._emit(
                    "pulse_started",
                    observation_sequence=observation.sequence,
                    waypoint_index=0,
                )
                # stream_move expires and calls StopMove itself.  Do not call
                # /v1/stop between pulses because it revokes operator-ready.
                await asyncio.sleep(duration_s + self.config.settle_s)
            elif self.config.execute:
                await asyncio.sleep(self.config.settle_s)

    async def _force_stop(self) -> None:
        if not self._motion_authorized:
            return
        await asyncio.shield(asyncio.to_thread(self.control.stop))
        self._motion_authorized = False

    async def run(
        self, instruction: str, *, episode_id: str | None = None
    ) -> Go2NavigationSessionResult:
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
        self._last_sequence = None
        self._inference_count = 0
        self._plans_accepted = 0
        self._motion_commands = 0
        self._motion_authorized = False
        reason: NavigationEndReason | None = None
        self._emit("session_started", message=f"execute={self.config.execute} reactive=true")
        try:
            if self.config.execute:
                try:
                    await asyncio.to_thread(self.control.preflight)
                except BaseException as error:
                    self._emit("preflight_failed", message=repr(error))
                    raise
                self._motion_authorized = True
                self._emit("preflight_passed")
            try:
                reason = await asyncio.wait_for(
                    self._cycle(instruction.strip(), episode_id),
                    timeout=self.config.max_runtime_s,
                )
            except asyncio.TimeoutError:
                reason = "max_runtime"
                self._emit("max_runtime")
        except BaseException as error:
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

        if reason is None:
            raise Go2NavigationSessionError("navigation ended without a terminal reason")
        self._emit("session_stopped", message=reason)
        return Go2NavigationSessionResult(
            episode_id=episode_id,
            reason=reason,
            elapsed_s=time.monotonic() - self._started_at_s,
            inference_count=self._inference_count,
            plans_accepted=self._plans_accepted,
            control_ticks=0,
            motion_commands=self._motion_commands,
            last_observation_sequence=self._last_sequence,
            events=tuple(self._events),
            events_dropped=self._events_dropped,
        )


__all__ = [
    "DEFAULT_GO2_REACTIVE_SESSION_CONFIG",
    "Go2ReactiveNavigationSession",
    "Go2ReactiveSessionConfig",
    "waypoint_to_velocity_pulse",
]
