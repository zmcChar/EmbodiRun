"""Assemble task sessions from a navigation policy and Go2 adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from embodied_runtime.policies.navigation.navila import (
    MAX_NAVILA_FORWARD_M,
    MAX_NAVILA_TURN_RAD,
)
from embodied_runtime.tasks.navigation import (
    NavigationSession,
    NavigationSessionConfig,
    NavigationSessionEvent,
    PlanarVelocityLimits,
    ReactiveNavigationSession,
    ReactiveNavigationSessionConfig,
    WaypointFollowerConfig,
    WorldWaypointFollower,
)

from .interfaces import ManagedNavigationPolicy
from .settings import Go2NavigationAppConfig


def resolve_session_mode(config: Go2NavigationAppConfig) -> str:
    mode = config.run.session_mode
    if mode == "auto":
        return "reactive" if config.policy.backend in {"streamvln", "navila"} else "continuous"
    return mode


def build_session(
    config: Go2NavigationAppConfig,
    policy: ManagedNavigationPolicy,
    camera: object,
    control: object,
    *,
    execute: bool,
    event_sink: Callable[[NavigationSessionEvent], None],
    session_factory: Callable[..., Any] | None = None,
) -> tuple[object, str]:
    limits = PlanarVelocityLimits(
        config.go2.max_abs_vx_mps,
        config.go2.max_abs_vy_mps,
        config.go2.max_abs_yaw_rate_rps,
    )
    mode = resolve_session_mode(config)
    if mode == "reactive":
        linear_speed_mps = min(0.30, limits.max_abs_vx_mps, limits.max_abs_vy_mps)
        yaw_rate_rps = min(0.60, limits.max_abs_yaw_rate_rps)
        max_pulse_s = 1.50
        if config.policy.backend == "navila":
            # Preserve NaVILA's largest native action after applying Go2's lower
            # velocity ceiling instead of silently truncating a 75 cm command.
            max_pulse_s = max(
                max_pulse_s,
                MAX_NAVILA_FORWARD_M / linear_speed_mps,
                MAX_NAVILA_TURN_RAD / yaw_rate_rps,
            )
        session_class = session_factory or ReactiveNavigationSession
        session = session_class(
            policy,
            camera,
            control,
            config=ReactiveNavigationSessionConfig(
                max_runtime_s=config.run.max_runtime_s,
                execute=execute,
                linear_speed_mps=linear_speed_mps,
                yaw_rate_rps=yaw_rate_rps,
                max_pulse_s=max_pulse_s,
                max_events=config.run.max_events,
                limits=limits,
            ),
            event_sink=event_sink,
        )
        return session, mode

    follower = WorldWaypointFollower(
        WaypointFollowerConfig(
            position_tolerance_m=config.go2.position_tolerance_m,
            yaw_tolerance_rad=config.go2.yaw_tolerance_rad,
            limits=limits,
        )
    )
    session_class = session_factory or NavigationSession
    session = session_class(
        policy,
        camera,
        control,
        follower=follower,
        config=NavigationSessionConfig(
            control_hz=config.run.control_hz,
            max_runtime_s=config.run.max_runtime_s,
            execute=execute,
            lease_duration_s=config.run.lease_duration_s,
            max_events=config.run.max_events,
        ),
        event_sink=event_sink,
    )
    return session, mode


__all__ = ["build_session", "resolve_session_mode"]
