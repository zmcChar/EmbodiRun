"""Unitree Go2 camera, control, and waypoint-tracking boundary.

Keep host-side navigation modules lazy: the nested ``agent`` package is also
deployed to the Go2's Python 3.8 environment and must not import them at boot.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_GO2_LIMITS",
    "DEFAULT_GO2_NAVIGATION_SESSION_CONFIG",
    "DEFAULT_WAYPOINT_FOLLOWER_CONFIG",
    "MAX_PROJECTION_S",
    "BaseVelocityCommand",
    "FollowerSample",
    "Go2CameraClient",
    "Go2CameraError",
    "Go2ClientError",
    "Go2ControlClient",
    "Go2Limits",
    "Go2NavigationSession",
    "Go2NavigationSessionConfig",
    "Go2NavigationSessionError",
    "Go2NavigationSessionResult",
    "Go2State",
    "Go2VelocityLease",
    "NavigationEndReason",
    "NavigationSessionEvent",
    "WaypointFollowerConfig",
    "WorldWaypointFollower",
    "capture_pose",
    "project_pose_to_time",
    "relative_target_to_velocity",
]

_SYMBOL_MODULES = {
    "Go2CameraClient": ".camera",
    "Go2CameraError": ".camera",
    "Go2ClientError": ".client",
    "Go2ControlClient": ".client",
    "Go2VelocityLease": ".motion",
    "DEFAULT_GO2_NAVIGATION_SESSION_CONFIG": ".session",
    "Go2NavigationSession": ".session",
    "Go2NavigationSessionConfig": ".session",
    "Go2NavigationSessionError": ".session",
    "Go2NavigationSessionResult": ".session",
    "NavigationEndReason": ".session",
    "NavigationSessionEvent": ".session",
    "MAX_PROJECTION_S": ".time_sync",
    "capture_pose": ".time_sync",
    "project_pose_to_time": ".time_sync",
    "DEFAULT_GO2_LIMITS": ".types",
    "BaseVelocityCommand": ".types",
    "Go2Limits": ".types",
    "Go2State": ".types",
    "DEFAULT_WAYPOINT_FOLLOWER_CONFIG": ".waypoint_follower",
    "FollowerSample": ".waypoint_follower",
    "WaypointFollowerConfig": ".waypoint_follower",
    "WorldWaypointFollower": ".waypoint_follower",
    "relative_target_to_velocity": ".waypoint_follower",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
