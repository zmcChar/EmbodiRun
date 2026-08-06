"""Unitree Go2 camera, control, and waypoint-tracking boundary."""

from .camera import Go2CameraClient, Go2CameraError
from .client import Go2ClientError, Go2ControlClient
from .motion import Go2VelocityLease
from .time_sync import MAX_PROJECTION_S, capture_pose, project_pose_to_time
from .types import DEFAULT_GO2_LIMITS, BaseVelocityCommand, Go2Limits, Go2State
from .waypoint_follower import (
    DEFAULT_WAYPOINT_FOLLOWER_CONFIG,
    FollowerSample,
    WaypointFollowerConfig,
    WorldWaypointFollower,
    relative_target_to_velocity,
)

__all__ = [
    "DEFAULT_GO2_LIMITS",
    "DEFAULT_WAYPOINT_FOLLOWER_CONFIG",
    "MAX_PROJECTION_S",
    "BaseVelocityCommand",
    "FollowerSample",
    "Go2CameraClient",
    "Go2CameraError",
    "Go2ClientError",
    "Go2ControlClient",
    "Go2Limits",
    "Go2State",
    "Go2VelocityLease",
    "WaypointFollowerConfig",
    "WorldWaypointFollower",
    "capture_pose",
    "project_pose_to_time",
    "relative_target_to_velocity",
]
