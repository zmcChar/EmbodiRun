"""Project Go2 odometry to a camera exposure timestamp."""

from __future__ import annotations

import math

from embodied_runtime.utils import Pose2D

from .types import Go2State

MAX_PROJECTION_S = 1.0


def project_pose_to_time(state: Go2State, captured_at_s: float) -> Pose2D:
    timestamp = state.received_at_s
    if timestamp is None:
        return state.pose
    delta_s = float(captured_at_s) - timestamp
    if not math.isfinite(delta_s) or abs(delta_s) > MAX_PROJECTION_S:
        return state.pose
    yaw_mid = state.pose.yaw_rad + 0.5 * state.yaw_rate_rps * delta_s
    world_vx = state.forward_velocity_mps * math.cos(
        yaw_mid
    ) - state.lateral_velocity_mps * math.sin(yaw_mid)
    world_vy = state.forward_velocity_mps * math.sin(
        yaw_mid
    ) + state.lateral_velocity_mps * math.cos(yaw_mid)
    return Pose2D(
        state.pose.x_m + world_vx * delta_s,
        state.pose.y_m + world_vy * delta_s,
        state.pose.yaw_rad + state.yaw_rate_rps * delta_s,
    )


def capture_pose(before: Go2State, after: Go2State, captured_at_s: float) -> Pose2D:
    timestamped = tuple(state for state in (before, after) if state.received_at_s is not None)
    if not timestamped:
        return after.pose
    nearest = min(
        timestamped,
        key=lambda state: abs(float(state.received_at_s) - captured_at_s),
    )
    return project_pose_to_time(nearest, captured_at_s)


__all__ = ["MAX_PROJECTION_S", "capture_pose", "project_pose_to_time"]
