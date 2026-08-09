"""Run one closed-loop navigation application composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from embodied_runtime.robots.unitree.go2 import Go2CameraClient, Go2ControlClient

from .composition import build_session
from .interfaces import ManagedNavigationPolicy
from .output import emit, json_value
from .policies import build_navigation_policy
from .settings import Go2NavigationAppConfig, PolicySettings

PolicyFactory = Callable[[PolicySettings], ManagedNavigationPolicy]


async def run_navigation(
    config: Go2NavigationAppConfig,
    *,
    execute: bool = False,
    output: Callable[[str], None] = print,
    policy_factory: PolicyFactory = build_navigation_policy,
    camera_factory: Callable[..., Any] = Go2CameraClient,
    control_factory: Callable[..., Any] = Go2ControlClient,
    session_factory: Callable[..., Any] | None = None,
) -> object:
    """Build and run one session. Motion is disabled unless ``execute`` is true."""

    if config.run.instruction is None:
        raise ValueError("navigation instruction is required (use --instruction)")
    policy = policy_factory(config.policy)
    try:
        # Prepare the selected runtime before the task session starts so local
        # loading/warmup or an external service handshake does not consume the
        # navigation runtime budget.
        await policy.prepare()
        camera = camera_factory(
            config.go2.camera_url,
            token=config.go2.camera_token,
            timeout_s=config.go2.camera_timeout_s,
        )
        control = control_factory(
            config.go2.control_url,
            token=config.go2.control_token,
            timeout_s=config.go2.control_timeout_s,
        )

        def event_sink(event: object) -> None:
            payload = json_value(event)
            if not isinstance(payload, Mapping):
                payload = {"event": payload}
            emit(output, payload)

        session, mode = build_session(
            config,
            policy,
            camera,
            control,
            execute=execute,
            event_sink=event_sink,
            session_factory=session_factory,
        )
        emit(
            output,
            {
                "kind": "navigation_start",
                "model": config.policy.model,
                "runtime": config.policy.runtime,
                # Compatibility field for existing JSON Lines consumers.
                "backend": config.policy.backend,
                "episode_id": config.run.episode_id,
                "mode": "execute" if execute else "dry-run",
                "session_mode": mode,
            },
        )
        result = await session.run(config.run.instruction, episode_id=config.run.episode_id)
    finally:
        await policy.aclose()

    summary = json_value(result)
    if not isinstance(summary, Mapping):
        summary = {"result": summary}
    summary = dict(summary)
    summary.pop("events", None)
    emit(
        output,
        {
            "kind": "navigation_summary",
            "ok": True,
            "model": config.policy.model,
            "runtime": config.policy.runtime,
            # Compatibility field for existing JSON Lines consumers.
            "backend": config.policy.backend,
            "mode": "execute" if execute else "dry-run",
            **summary,
        },
    )
    return result


__all__ = ["PolicyFactory", "run_navigation"]
